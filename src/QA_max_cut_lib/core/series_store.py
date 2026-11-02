"""Write-once storage and read-back for experiment records."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from .experiment_ids import stable_hash
from .records import to_json_data
from .saving import publish_files, utc_now, _name

__all__ = ["ExperimentStore", "load_saved_records", "load_saved_series"]


class ExperimentStore:
    """Publish trial, reference, and manifest files without overwrite.

    Parameters
    ----------
    config : dict
        Save configuration with ``path``, ``filename``, ``makedir``, and
        ``savedir``.  ``atts`` and ``save_original_data`` are accepted for
        compatibility with the common exporter; series records contain no
        native arrays in this milestone.

    Notes
    -----
    The coordinator is the only writer.  Workers return records; they do
    not append to a shared CSV or directory.
    """

    def __init__(self, config, *, default_filename="experiment_series"):
        if config is None:
            self.enabled = False
            self.folder = None
            self.written = []
            self.reused = []
            return
        config = deepcopy(config)
        allowed = {"path", "filename", "makedir", "savedir", "atts", "save_original_data", "format", "DPI", "metadata"}
        if set(config) - allowed:
            raise ValueError(f"Unknown series save field: {sorted(set(config) - allowed)}")
        if "path" not in config:
            raise ValueError("series save.path is required.")
        self.enabled = True
        self.base = Path(config["path"])
        self.stem = _name(config.get("filename", default_filename))
        self.makedir = bool(config.get("makedir", False))
        self.savedir = bool(config.get("savedir", False))
        for name in ("atts", "save_original_data"):
            if name in config and not isinstance(config[name], list):
                raise ValueError(f"series save.{name} must be a list.")
        if not self.base.exists():
            if not self.makedir:
                raise FileNotFoundError(self.base)
            self.base.mkdir(parents=True, exist_ok=True)
        if not self.base.is_dir():
            raise NotADirectoryError(self.base)
        self.folder = self.base / self.stem if self.savedir else self.base
        self.folder.mkdir(parents=True, exist_ok=True)
        self.written = []
        self.reused = []

    def _write_json(self, name, document):
        payload = json.dumps(
            document, ensure_ascii=False, allow_nan=False, indent=2
        ).encode("utf-8")
        files = publish_files(
            {
                "path": str(self.folder),
                "filename": name,
                "makedir": True,
                "savedir": False,
            },
            {name: payload},
            name,
        )
        self.written.append(str(files.primary))
        return files.primary

    def write_record(self, record):
        """Publish one completed, stopped, or failed record.

        Existing files with the same full identity are reused only when the
        complete attempt identity matches.  A short-name collision with another
        full id or a new attempt receives a distinct write-once filename.
        """
        if not self.enabled:
            return None
        identity = record.get("trial_id") or record.get("reference_id")
        if not identity:
            raise ValueError("A stored record needs trial_id or reference_id.")
        prefix = "trial" if record.get("record_type") == "trial" else "reference"
        base = f"{prefix}_{str(identity)[:16]}"
        attempt_tag = str(record.get("attempt_id", ""))[:10]
        candidates = [base + ".json"]
        if attempt_tag:
            candidates.append(f"{base}_{attempt_tag}.json")
        for version in range(2, 10000):
            candidates.append(f"{base}_{version}.json")
        for name in candidates:
            target = self.folder / name
            if not target.exists():
                document = {
                    "schema_version": 1,
                    "export_type": "experiment_record",
                    "export_complete": True,
                    "saved_at": utc_now(),
                    "record_type": record.get("record_type"),
                    "record": to_json_data(record),
                }
                return self._write_json(name, document)
            previous = json.loads(target.read_text(encoding="utf-8"))
            previous_record = previous.get("record", {})
            key = "trial_id" if prefix == "trial" else "reference_id"
            if previous_record.get(key) == identity and previous_record.get("attempt_id") == record.get("attempt_id"):
                self.reused.append(str(target))
                return target
        raise FileExistsError(f"Could not allocate a record name for {identity}.")

    def write_manifest(self, document):
        """Publish a summary manifest, versioning rather than overwriting."""
        if not self.enabled:
            return None
        if not isinstance(document, dict):
            raise TypeError("manifest must be a mapping.")
        for version in range(1, 10000):
            suffix = "" if version == 1 else f"_v{version}"
            name = f"{self.stem}_summary{suffix}.json"
            target = self.folder / name
            if not target.exists():
                payload = {
                    "schema_version": 1,
                    "export_type": "experiment_manifest",
                    "export_complete": True,
                    "saved_at": utc_now(),
                    "manifest": to_json_data(document),
                }
                return self._write_json(name, payload)
        raise FileExistsError("Could not allocate a manifest version.")

    def snapshot(self):
        return {
            "enabled": self.enabled,
            "folder": None if self.folder is None else str(self.folder),
            "written": list(self.written),
            "reused": list(self.reused),
        }

    def read_trial_records(self):
        if not self.enabled:
            return {}
        result = {}
        for path in sorted(self.folder.glob("trial_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("export_type") != "experiment_record" or not data.get("export_complete"):
                continue
            record = data.get("record", {})
            identity = record.get("trial_id")
            if identity:
                result[identity] = record
        return result

    def read_reference_records(self):
        if not self.enabled:
            return {}
        result = {}
        for path in sorted(self.folder.glob("reference_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("export_type") != "experiment_record" or not data.get("export_complete"):
                continue
            record = data.get("record", {})
            identity = record.get("reference_id")
            if identity:
                result[identity] = record
        return result


def _iter_records(path):
    path = Path(path)
    if path.is_dir():
        for name in ("trial_*.json", "reference_*.json"):
            for file in sorted(path.glob(name)):
                data = json.loads(file.read_text(encoding="utf-8"))
                if data.get("export_complete"):
                    yield data
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("export_complete"):
            yield data


def load_saved_records(path):
    """Read completed trial and reference records without running solvers."""
    trials, references = [], []
    for data in _iter_records(path):
        record = data.get("record", {})
        if data.get("record_type") == "trial":
            trials.append(record)
        elif data.get("record_type") == "reference":
            references.append(record)
    return {"trials": trials, "references": references}


def load_saved_series(path):
    """Read a saved manifest or compose one from completed records."""
    path = Path(path)
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("export_type") == "experiment_manifest" and data.get("export_complete"):
            return data["manifest"]
        if data.get("export_type") == "experiment_record":
            return {"trials": [data["record"]], "references": []}
        raise ValueError("Not a complete experiment manifest.")
    return load_saved_records(path)
