"""Atomic JSON/NPZ checkpoint storage and resumable run controls.

The manager stores portable scientific state, not live solver objects.  It
uses a strict JSON manifest plus numeric ``.npy`` payloads, one immutable
generation per commit, and a small latest-index.  A single process owns one
managed run at a time.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np

from .experiment_ids import stable_hash
from .._version import __version__

__all__ = [
    "RunControl",
    "CheckpointError",
    "CheckpointCorrupt",
    "CheckpointIncompatible",
    "CheckpointLocked",
    "CheckpointRef",
    "CheckpointManager",
    "ResourceJournal",
    "inspect_checkpoint",
    "restore",
    "load_checkpoint",
    "validate_checkpoint_config",
]


def _now():
    return datetime.now(timezone.utc).isoformat()


class CheckpointError(RuntimeError):
    """Base checkpoint failure."""


class CheckpointCorrupt(CheckpointError):
    """The manifest or a payload failed integrity checks."""


class CheckpointIncompatible(CheckpointError):
    """The stored state cannot be continued under the requested policy."""


class CheckpointLocked(CheckpointError):
    """Another owner already holds this managed run."""


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Checkpoint state contains a non-finite float.")
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"Checkpoint state contains unsupported {type(value).__name__}.")


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _fsync_path(path):
    try:
        with open(path, "rb") as stream:
            os.fsync(stream.fileno())
    except OSError:
        pass


class RunControl:
    """Explicit pause/cancel control for one managed run.

    Parameters
    ----------
    pause_after : dict, optional
        Event name to absolute completed count.  The supported QAOACPU event
        is ``optimizer_iterations``.
    request_pause : bool, default False
        Pause at the next supported boundary, independently of counters.
    """

    def __init__(self, pause_after=None, *, request_pause=False):
        if pause_after is None:
            pause_after = {}
        if not isinstance(pause_after, dict):
            raise TypeError("pause_after must be a mapping.")
        self._pause_after = {}
        for key, value in pause_after.items():
            if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("pause_after needs nonnegative integer event counters.")
            self._pause_after[key] = int(value)
        self._requested = bool(request_pause)
        self._consumed = {}
        self._cancel_requested = False

    def request_pause(self):
        """Ask the algorithm to pause at its next supported boundary."""
        self._requested = True

    def request_cancel(self):
        """Ask the algorithm to stop at the next supported boundary."""
        self._cancel_requested = True

    def should_pause(self, event, completed):
        """Return True once for a reached trigger.

        An absolute trigger is marked consumed when it fires, so a resumed
        run cannot immediately fire on the same historical counter.
        """
        if self._cancel_requested:
            return True
        value = int(completed)
        trigger = self._pause_after.get(event)
        if trigger is not None and value >= trigger and self._consumed.get(event) is not True:
            self._consumed[event] = True
            return True
        if self._requested and not self._consumed.get("request", False):
            self._consumed["request"] = True
            return True
        return False

    def snapshot(self):
        return {
            "pause_after": deepcopy(self._pause_after),
            "requested": bool(self._requested),
            "cancel_requested": bool(self._cancel_requested),
            "consumed": deepcopy(self._consumed),
        }

    @classmethod
    def from_snapshot(cls, snapshot):
        control = cls((snapshot or {}).get("pause_after", {}), request_pause=bool((snapshot or {}).get("requested", False)))
        control._consumed = deepcopy((snapshot or {}).get("consumed", {}))
        control._cancel_requested = bool((snapshot or {}).get("cancel_requested", False))
        return control


class ResourceJournal:
    """Append-only durable reservation journal for one managed run.

    A BEGIN record is never silently released.  If no COMMIT/resolution exists,
    recovery charges its upper bound once.  The journal is deliberately small
    and uses one owner.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "resource_journal.jsonl"
        self._seq = 0

    def _append(self, record):
        self._seq += 1
        record = {"seq": self._seq, "time": _now(), **record}
        data = (json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        with self.path.open("ab") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return record

    def begin(self, operation_id, *, attempt_id, upper_bound, kind="operation"):
        if upper_bound is None:
            raise ValueError("A durable reservation needs a finite upper_bound.")
        return self._append(
            {
                "event": "BEGIN",
                "operation_id": str(operation_id),
                "attempt_id": str(attempt_id),
                "upper_bound": int(upper_bound),
                "kind": str(kind),
            }
        )

    def commit(self, operation_id, *, used, receipt=None):
        return self._append(
            {
                "event": "COMMIT",
                "operation_id": str(operation_id),
                "used": int(used),
                "receipt": deepcopy(receipt) if receipt is not None else None,
            }
        )

    def resolve(self, operation_id, *, charge, reason):
        return self._append(
            {
                "event": "RESOLVE",
                "operation_id": str(operation_id),
                "charge": int(charge),
                "reason": str(reason),
            }
        )

    def read(self):
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise CheckpointCorrupt(f"Broken journal tail: {error}") from error
        return records

    def reconcile(self):
        """Return known, uncertain, and open reservations.

        A valid BEGIN without COMMIT/RESOLVE remains uncertain and is charged
        at its upper bound.  Repeating reconciliation does not charge it twice
        because the resolved operation id is remembered in the returned data.
        """
        begins = {}
        committed = {}
        resolved = {}
        for record in self.read():
            event = record.get("event")
            oid = record.get("operation_id")
            if event == "BEGIN":
                begins[oid] = record
            elif event == "COMMIT":
                committed[oid] = record
            elif event == "RESOLVE":
                resolved[oid] = record
        known = 0
        uncertain = 0
        open_ids = []
        for oid, begin in begins.items():
            if oid in committed:
                known += int(committed[oid].get("used", 0))
            elif oid in resolved:
                known += int(resolved[oid].get("charge", 0))
            else:
                uncertain += int(begin.get("upper_bound", 0))
                open_ids.append(oid)
        return {
            "known_spent": known,
            "uncertain_upper_charge": uncertain,
            "total_charged": known + uncertain,
            "open_operation_ids": open_ids,
            "complete": True,
        }


@dataclass(frozen=True)
class CheckpointRef:
    """Describe one published generation."""

    path: Path
    generation: int
    run_id: str
    manifest_path: Path

    def as_dict(self):
        return {"path": str(self.path), "generation": int(self.generation), "run_id": self.run_id}


def validate_checkpoint_config(config):
    """Validate and resolve a solver checkpoint section."""
    config = deepcopy(config or {})
    if not config:
        return {"enabled": False}
    allowed = {
        "enabled",
        "path",
        "format",
        "every",
        "at_phase_boundaries",
        "on_pause",
        "durable_resource_journal",
        "durability",
        "on_write_error",
        "retention",
    }
    if set(config) - allowed:
        raise ValueError(f"Unknown checkpoint fields: {sorted(set(config) - allowed)}")
    if "enabled" in config and type(config["enabled"]) is not bool:
        raise ValueError("checkpoint.enabled must be boolean.")
    enabled = bool(config.get("enabled", False))
    if not enabled:
        return {"enabled": False}
    if not config.get("path"):
        raise ValueError("checkpoint.path is required when checkpoint is enabled.")
    config.setdefault("format", "json_npy_v1")
    if config["format"] != "json_npy_v1":
        raise ValueError("Only json_npy_v1 checkpoints are implemented.")
    every = config.setdefault("every", {})
    if not isinstance(every, dict) or set(every) - {"optimizer_iterations", "reductions", "levels", "local_tasks"}:
        raise ValueError("Unknown checkpoint.every event.")
    for key, value in every.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("checkpoint.every values must be positive integers.")
    for key in ("at_phase_boundaries", "on_pause", "durable_resource_journal"):
        config.setdefault(key, True)
        if type(config[key]) is not bool:
            raise ValueError(f"checkpoint.{key} must be boolean.")
    config.setdefault("durability", "local_fsync")
    if config["durability"] not in ("local_fsync", "best_effort"):
        raise ValueError("checkpoint.durability must be local_fsync or best_effort.")
    config.setdefault("on_write_error", "stop")
    if config["on_write_error"] not in ("stop", "raise"):
        raise ValueError("checkpoint.on_write_error must be stop or raise.")
    retention = config.setdefault("retention", {"type": "keep_all"})
    if not isinstance(retention, dict) or retention.get("type") != "keep_all":
        raise ValueError("Only retention.type='keep_all' is implemented.")
    config.setdefault("path", str(config["path"]))
    return config


_ACTIVE = set()


class CheckpointManager:
    """Manage immutable generations for one canonical run."""

    def __init__(self, config, *, run_id, component_id, graph=None, config_signature=None):
        self.config = validate_checkpoint_config(config)
        self.enabled = bool(self.config.get("enabled", False))
        self.run_id = str(run_id)
        self.component_id = str(component_id)
        self.root = Path(self.config["path"]).expanduser() / self.run_id if self.enabled else None
        self.graph = graph
        self.config_signature = config_signature
        self.last_ref = None
        self._lock_handle = None
        self._locked = False
        self.journal = None
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            self.journal = ResourceJournal(self.root) if self.config.get("durable_resource_journal", True) else None
            self._load_latest_ref()

    def _load_latest_ref(self):
        latest = self.root / "latest.json"
        if not latest.exists():
            self.last_ref = None
            return None
        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
            ref = self.get(int(data["generation"]))
            self.last_ref = ref
            return ref
        except Exception as error:
            raise CheckpointCorrupt(f"Cannot read latest checkpoint: {error}") from error

    def acquire(self):
        """Take one owner lock for this managed run."""
        if not self.enabled or self._locked:
            return
        key = str(self.root.resolve())
        if key in _ACTIVE:
            raise CheckpointLocked(f"Run is already active: {self.run_id}")
        lock_path = self.root / "owner.lock"
        handle = open(lock_path, "a+", encoding="utf-8")
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception as error:
            handle.close()
            raise CheckpointLocked(f"Cannot lock {self.run_id}: {error}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        os.fsync(handle.fileno())
        self._lock_handle = handle
        self._locked = True
        _ACTIVE.add(key)

    def release(self):
        if not self._locked:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._lock_handle.seek(0)
                msvcrt.locking(self._lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            self._lock_handle.close()
        except Exception:
            pass
        self._lock_handle = None
        self._locked = False
        _ACTIVE.discard(str(self.root.resolve()))

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_):
        self.release()

    def list_generations(self):
        if not self.enabled or not self.root.exists():
            return []
        result = []
        for path in sorted(self.root.glob("gen_*")):
            if path.is_dir() and (path / "manifest.json").exists():
                try:
                    result.append(int(path.name.split("_", 1)[1]))
                except ValueError:
                    pass
        return sorted(result)

    def get(self, generation):
        path = self.root / f"gen_{int(generation):06d}"
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise CheckpointCorrupt(f"Checkpoint generation does not exist: {generation}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return CheckpointRef(path=path, generation=int(generation), run_id=manifest.get("run_id", self.run_id), manifest_path=manifest_path)

    def _next_generation(self):
        existing = self.list_generations()
        return (max(existing) + 1) if existing else 1

    def write_generation(self, state, arrays=None, *, parent=None, phase=None, next_operation=None):
        """Publish one complete generation and atomically update latest."""
        if not self.enabled:
            return None
        self.acquire()
        arrays = {} if arrays is None else dict(arrays)
        generation = self._next_generation()
        final = self.root / f"gen_{generation:06d}"
        if final.exists():
            raise CheckpointError(f"Generation already exists: {final}")
        staging = self.root / f".staging_{generation:06d}_{uuid.uuid4().hex}"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            state = deepcopy(state)
            state["generation"] = generation
            state["run_id"] = self.run_id
            payloads = []
            for name, array in arrays.items():
                if not isinstance(name, str) or "/" in name or "\\" in name or name.startswith("."):
                    raise ValueError("Invalid array payload name.")
                value = np.asarray(array)
                if value.dtype.hasobject:
                    raise TypeError("Object arrays are not checkpoint payloads.")
                filename = f"{name}.npy"
                path = staging / filename
                with path.open("wb") as stream:
                    np.save(stream, value, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                payloads.append(
                    {
                        "name": name,
                        "file": filename,
                        "dtype": value.dtype.str,
                        "shape": list(value.shape),
                        "size": path.stat().st_size,
                        "sha256": _digest(path.read_bytes()),
                    }
                )
            manifest = {
                "schema_version": 1,
                "format": "json_npy_v1",
                "checkpoint_id": stable_hash(
                    {"run_id": self.run_id, "generation": generation, "parent": parent}
                ),
                "run_id": self.run_id,
                "generation": generation,
                "parent_generation": parent,
                "component_id": self.component_id,
                "package_version": __version__,
                "created_at": _now(),
                "phase": phase,
                "next_operation": deepcopy(next_operation),
                "config_signature": self.config_signature,
                "graph_fingerprint": None
                if self.graph is None
                else self.graph.snapshot().get("fingerprint"),
                "array_payloads": payloads,
                "journal_cutoff": None if self.journal is None else self.journal.reconcile(),
                "state_file": "state.json",
            }
            state_path = staging / "state.json"
            state_data = json.dumps(_json(state), ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2).encode("utf-8")
            with state_path.open("wb") as stream:
                stream.write(state_data)
                stream.flush()
                os.fsync(stream.fileno())
            manifest["state_sha256"] = _digest(state_data)
            manifest_path = staging / "manifest.json"
            manifest_data = json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2).encode("utf-8")
            with manifest_path.open("wb") as stream:
                stream.write(manifest_data)
                stream.flush()
                os.fsync(stream.fileno())
            if self.config["durability"] == "local_fsync":
                _fsync_path(staging)
            os.replace(staging, final)
            if self.config["durability"] == "local_fsync":
                _fsync_path(self.root)
            latest = self.root / "latest.json"
            temp_latest = self.root / f".latest_{uuid.uuid4().hex}.tmp"
            with temp_latest.open("w", encoding="utf-8") as stream:
                json.dump({"generation": generation, "checkpoint_id": manifest["checkpoint_id"]}, stream, ensure_ascii=False, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_latest, latest)
            ref = CheckpointRef(path=final, generation=generation, run_id=self.run_id, manifest_path=final / "manifest.json")
            self.last_ref = ref
            return ref
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def load(self, path=None, *, verify=True):
        """Read one generation, or the latest generation when path is omitted."""
        if not self.enabled and path is None:
            raise CheckpointError("Checkpoints are disabled.")
        if path is None:
            if self.last_ref is None:
                self._load_latest_ref()
            if self.last_ref is None:
                raise CheckpointCorrupt("No checkpoint generation exists.")
            path = self.last_ref.path
        path = Path(path)
        if path.is_file():
            path = path.parent
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise CheckpointCorrupt(f"Missing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "json_npy_v1":
            raise CheckpointIncompatible("Unsupported checkpoint format.")
        state_path = path / manifest.get("state_file", "state.json")
        if not state_path.exists():
            raise CheckpointCorrupt("Missing state.json.")
        state_data = state_path.read_bytes()
        if manifest.get("state_sha256") and _digest(state_data) != manifest["state_sha256"]:
            raise CheckpointCorrupt("state.json checksum mismatch.")
        state = json.loads(state_data.decode("utf-8"))
        arrays = {}
        for payload in manifest.get("array_payloads", []):
            file = path / payload["file"]
            if not file.exists():
                raise CheckpointCorrupt(f"Missing array payload: {payload['file']}")
            if file.stat().st_size != payload.get("size"):
                raise CheckpointCorrupt(f"Array size mismatch: {payload['file']}")
            if verify and _digest(file.read_bytes()) != payload.get("sha256"):
                raise CheckpointCorrupt(f"Array checksum mismatch: {payload['file']}")
            arrays[payload["name"]] = np.load(file, allow_pickle=False)
        return {"manifest": manifest, "state": state, "arrays": arrays, "path": path}


def load_checkpoint(path):
    """Load and verify one checkpoint path using a temporary manager."""
    path = Path(path)
    if path.is_file() and path.name == "latest.json":
        data = json.loads(path.read_text(encoding="utf-8"))
        path = path.parent / f"gen_{int(data['generation']):06d}"
    elif path.is_dir() and not (path / "manifest.json").exists() and (path / "latest.json").exists():
        data = json.loads((path / "latest.json").read_text(encoding="utf-8"))
        path = path / f"gen_{int(data['generation']):06d}"
    manager = CheckpointManager({"enabled": True, "path": str(path.parent)}, run_id=path.name, component_id="inspect")
    # The generation directory may be nested under a run root.
    try:
        return manager.load(path)
    finally:
        manager.release()


def inspect_checkpoint(path):
    """Read manifest metadata without constructing a solver or CUDA context."""
    loaded = load_checkpoint(path)
    manifest = loaded["manifest"]
    return {
        "schema_version": 1,
        "checkpoint_id": manifest.get("checkpoint_id"),
        "run_id": manifest.get("run_id"),
        "generation": manifest.get("generation"),
        "phase": manifest.get("phase"),
        "component_id": manifest.get("component_id"),
        "package_version": manifest.get("package_version"),
        "next_operation": manifest.get("next_operation"),
        "array_payloads": manifest.get("array_payloads", []),
        "journal_cutoff": manifest.get("journal_cutoff"),
        "path": str(loaded["path"]),
    }


def restore(path, *, overrides=None):
    """Restore a QAOACPU object bound to a checkpoint, without running it."""
    loaded = load_checkpoint(path)
    manifest = loaded["manifest"]
    state = loaded["state"]
    component = manifest.get("component_id")
    from .checkpoint_capabilities import capability

    if not capability(component).get("supports_checkpoint", False):
        raise CheckpointIncompatible(
            f"Component {component!r} has no checkpoint resume codec."
        )
    config = deepcopy(state.get("config"))
    if not isinstance(config, dict):
        raise CheckpointCorrupt("Checkpoint has no resolved solver config.")
    if manifest.get("config_signature"):
        expected = stable_hash({"config": state.get("config"), "graph": state.get("graph")})
        if expected != manifest["config_signature"]:
            raise CheckpointIncompatible("Checkpoint config or graph fingerprint changed.")
    if "checkpoint" in config:
        # Resolve the managed store relative to the actual restored tree, not
        # an absolute path from the machine that wrote the checkpoint.
        config["checkpoint"] = deepcopy(config["checkpoint"])
        config["checkpoint"]["path"] = str(loaded["path"].parent.parent)
    if overrides:
        allowed = {"output", "checkpoint"}
        if set(overrides) - allowed:
            raise CheckpointIncompatible(f"Unsupported restore overrides: {sorted(set(overrides) - allowed)}")
        for key, value in overrides.items():
            config[key] = deepcopy(value)
    from ..core.factory import SolverFactory

    solver = SolverFactory.from_config(config)
    graph = state.get("graph")
    if not isinstance(graph, dict):
        raise CheckpointCorrupt("Checkpoint has no graph snapshot.")
    from .graph import WeightedGraph

    solver._resume_state = deepcopy(state)
    solver._resume_arrays = {key: np.asarray(value) for key, value in loaded["arrays"].items()}
    solver._resume_run_id = manifest.get("run_id")
    solver._resume_graph = WeightedGraph.from_snapshot(graph)
    solver._resume_control = RunControl.from_snapshot(state.get("control"))
    solver.last_checkpoint = CheckpointRef(
        path=loaded["path"], generation=int(manifest["generation"]), run_id=manifest["run_id"], manifest_path=loaded["path"] / "manifest.json"
    )
    return solver
