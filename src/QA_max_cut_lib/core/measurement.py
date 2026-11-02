"""Common trial executor and measurement protocol."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
import platform
import sys
from time import perf_counter_ns

from .experiment_ids import graph_content_id, graph_instance_id, measurement_environment_id, stable_hash
from .graph import WeightedGraph
from .records import to_json_data

__all__ = [
    "measurement_environment",
    "estimate_trial_shots",
    "CampaignBudget",
    "execute_trial",
    "work_signature",
    "TrialExecutionError",
]


class TrialExecutionError(RuntimeError):
    """Report a solver call that did not produce a usable complete solution."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _safe_version(module_name):
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def measurement_environment(include_gpu=True):
    """Collect a JSON-compatible machine and library fingerprint.

    Physical core count is reported only when the operating system exposes
    it.  Unknown topology is represented by null, never by an invented
    number.
    """
    physical = None
    affinity = None
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        affinity = len(psutil.Process().cpu_affinity())
    except Exception:
        pass
    threadpools = None
    try:
        from threadpoolctl import threadpool_info

        threadpools = threadpool_info()
    except Exception:
        pass
    gpu = None
    if include_gpu:
        try:
            import cupy

            count = int(cupy.cuda.runtime.getDeviceCount())
            if count:
                props = cupy.cuda.runtime.getDeviceProperties(0)
                gpu = {
                    "backend": "cupy",
                    "cupy_version": getattr(cupy, "__version__", None),
                    "device_count": count,
                    "device_name": props.get("name", b"").decode(errors="replace")
                    if isinstance(props.get("name", b""), bytes)
                    else str(props.get("name", "")),
                    "compute_capability": f"{props.get('major', 0)}.{props.get('minor', 0)}",
                    "total_memory": int(props.get("totalGlobalMem", 0)),
                    "driver_version": int(cupy.cuda.runtime.driverGetVersion()),
                    "runtime_version": int(cupy.cuda.runtime.runtimeGetVersion()),
                }
        except Exception as error:
            gpu = {"backend": "cupy", "available": False, "reason": f"{type(error).__name__}: {error}"}
    data = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": physical,
        "cpu_affinity_count": affinity,
        "cpu_affinity": sorted(psutil.Process().cpu_affinity()) if affinity else None,
        "threadpool_info": threadpools,
        "packages": {
            name: _safe_version(name)
            for name in ("numpy", "scipy", "cvxpy", "scs", "dimod", "dwave.samplers", "cupy", "threadpoolctl", "psutil")
        },
        "gpu": gpu,
        "timer": "perf_counter_ns",
    }
    # Dynamic native-pool details are recorded for diagnosis but excluded from
    # the reusable environment identity: merely importing a numerical module
    # can change threadpoolctl's view without changing the machine.
    identity_fields = {
        key: value
        for key, value in data.items()
        if key not in {"threadpool_info", "cpu_affinity_count", "cpu_affinity"}
    }
    data["environment_id"] = measurement_environment_id(identity_fields)
    return data


def _shot_cap(config):
    """Return a finite shot upper bound, zero, or None if unknown."""
    if not isinstance(config, dict):
        return 0
    kind = config.get("type")
    if kind in ("exact_cpu", "simulated_annealing_cpu", "gw_cpu"):
        return 0
    limits = config.get("limits", {}) if isinstance(config.get("limits", {}), dict) else {}
    if "shots_total" in limits:
        return limits["shots_total"]
    if kind == "refined":
        base = config.get("base_solver", {})
        return _shot_cap(base)
    return None


def estimate_trial_shots(config):
    """Estimate the campaign shot reservation for one top-level trial.

    Returns
    -------
    int or None
        A finite upper bound, zero for classical methods, or None when the
        child solver has no finite root shot cap.
    """
    return _shot_cap(config)


class CampaignBudget:
    """Admission ledger for campaign runs and shots."""

    def __init__(self, *, max_runs=None, shots_total=None):
        for name, value in (("max_runs", max_runs), ("shots_total", shots_total)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer or None.")
        self.max_runs = max_runs
        self.shots_total = shots_total
        self.runs_reserved = 0
        self.runs_completed = 0
        self.shots_reserved = 0
        self.shots_used = 0
        self.open_reservations = []

    @property
    def runs_remaining(self):
        return None if self.max_runs is None else self.max_runs - self.runs_reserved

    def reserve_run(self, count=1):
        count = int(count)
        if count < 0:
            raise ValueError("count must be nonnegative.")
        if self.max_runs is not None and self.runs_reserved + count > self.max_runs:
            return False
        self.runs_reserved += count
        return True

    def release_run(self, count=1):
        self.runs_reserved = max(0, self.runs_reserved - int(count))

    def reserve_shots(self, amount):
        if amount is None:
            if self.shots_total is None:
                return True
            return False
        amount = int(amount)
        if amount < 0:
            raise ValueError("amount must be nonnegative.")
        if self.shots_total is not None and self.shots_reserved + amount > self.shots_total:
            return False
        self.shots_reserved += amount
        self.open_reservations.append(amount)
        return True

    def settle_shots(self, reserved, used):
        reserved = 0 if reserved is None else int(reserved)
        used = 0 if used is None else int(used)
        self.shots_reserved = max(0, self.shots_reserved - reserved)
        if self.open_reservations:
            for index in range(len(self.open_reservations) - 1, -1, -1):
                if self.open_reservations[index] == reserved:
                    self.open_reservations.pop(index)
                    break
        self.shots_used += used
        # A solver must not exceed its root cap.  If it does, expose the
        # overrun rather than silently pretending the ledger is exact.
        if self.shots_total is not None and self.shots_used > self.shots_total:
            raise RuntimeError("Campaign shot ledger exceeds its total.")

    def snapshot(self):
        return {
            "max_runs_total": self.max_runs,
            "runs_reserved": self.runs_reserved,
            "runs_completed": self.runs_completed,
            "shots_total": self.shots_total,
            "shots_reserved": self.shots_reserved,
            "shots_used": self.shots_used,
            "shots_remaining": None if self.shots_total is None else self.shots_total - self.shots_used,
            "open_reservations": list(self.open_reservations),
        }


def _has_gpu_work(config):
    text = json.dumps(config, ensure_ascii=False, sort_keys=True).lower()
    return "gpu" in text


def _synchronize_gpu(config, graph):
    if not _has_gpu_work(config):
        return False
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            return False
        cupy.cuda.runtime.deviceSynchronize()
        return True
    except Exception:
        return False


def work_signature(record):
    """Hash actual work counters, not merely a repeated config string."""
    resources = record.get("resources", {}) if isinstance(record, dict) else {}
    keys = (
        "states_evaluated",
        "objective_evaluations",
        "optimizer_steps",
        "training_shots",
        "readout_shots",
        "shots_used",
        "anneal_reads_completed",
        "anneal_sweeps_accounted",
        "reductions_completed",
        "level_count",
        "local_solver_calls",
        "tasks_total",
        "num_hyperplanes_completed",
        "refinement_states_evaluated",
    )
    payload = {
        "status": record.get("status") if isinstance(record, dict) else None,
        "termination_reason": record.get("termination_reason") if isinstance(record, dict) else None,
        "counters": {key: resources.get(key) for key in keys if key in resources},
    }
    return stable_hash({"schema": "work-v1", "work": payload}), payload


def _effective_threads(solver, requested):
    if requested == 1 or bool(getattr(solver, "supports_parallel", False)):
        return requested, None
    return 1, "implementation_sequential"


def execute_trial(plan_trial, *, attempt=0, protocol=None):
    """Execute one planned trial and return a structured trial record.

    Parameters
    ----------
    plan_trial : PlannedTrial
        Planned trial with a case and graph.
    attempt : int, default 0
        Attempt index.  Retries get a new id and additional cost.
    protocol : dict, optional
        Resolved measurement protocol.  The trial protocol is used when
        omitted.

    Returns
    -------
    dict
        A JSON-compatible complete or failed record.  A failed record has
        ``cut_value=None``; it never invents ``C=0``.

    Notes
    -----
    ``end_to_end`` includes solver construction, solve, and GPU
    synchronisation.  ``solve_only`` prepares the solver before the timer.
    The scientific config is never rewritten to improve timing.
    """
    from .factory import SolverFactory

    case = plan_trial.case
    graph = case.graph.copy()
    protocol = deepcopy(protocol or plan_trial.protocol)
    config = deepcopy(case.config)
    config["seed"] = int(plan_trial.seed)
    config.pop("save", None)
    config.pop("output", None)
    config["output"] = {
        "report": int(protocol.get("trial_output", {}).get("report", 0)),
        "logs": int(protocol.get("trial_output", {}).get("logs", 0)),
    }
    requested = int(plan_trial.num_treads)
    scope = protocol.get("timing_scope", "end_to_end")
    started_at = utc_now()
    t0 = perf_counter_ns()
    solver = None
    effective = requested
    capability_reason = None
    solver_compute_seconds = None
    resources = {}
    solution = None
    status = "failed"
    reason = None
    work = {"counters": {}}
    synced = False
    try:
        if scope == "solve_only":
            solver = SolverFactory.from_config(config)
            effective, capability_reason = _effective_threads(solver, requested)
            config["num_treads"] = effective
            if requested != effective:
                solver._config["num_treads"] = effective
            t_start = perf_counter_ns()
            solution = solver.solve(graph, report=0, logs=0, num_treads=effective)
            t_end = perf_counter_ns()
            wall_ns = t_end - t_start
        else:
            solver = SolverFactory.from_config(config)
            effective, capability_reason = _effective_threads(solver, requested)
            config["num_treads"] = effective
            if requested != effective:
                solver._config["num_treads"] = effective
            t_start = perf_counter_ns()
            solution = solver.solve(graph, report=0, logs=0, num_treads=effective)
            synced = _synchronize_gpu(config, graph) if protocol.get("synchronize_gpu", True) else False
            t_end = perf_counter_ns()
            wall_ns = t_end - t_start
            if synced:
                resources["gpu_synchronized"] = True
        record = solver.last_run
        if record is None:
            raise TrialExecutionError("Solver returned no run record.")
        solution = {int(key): int(value) for key, value in solution.items()}
        if set(solution) != set(graph.nodes):
            raise TrialExecutionError("Solver returned an incomplete assignment.")
        cut = graph.cut_value(solution)
        status = str(record.get("status", "completed"))
        reason = record.get("termination_reason")
        solver_compute_seconds = record.get("timing", {}).get("compute_seconds")
        resources = deepcopy(record.get("resources", {}))
        if synced:
            resources["gpu_synchronized"] = True
        work, work_detail = work_signature(record)
    except Exception as error:
        solution = None
        status = "failed"
        reason = f"{type(error).__name__}: {error}"
        wall_ns = perf_counter_ns() - t0
        cut = None
        solver = None
    if attempt < 0:
        raise ValueError("attempt must be nonnegative.")
    attempt_id = stable_hash(
        {
            "schema": "attempt-v1",
            "trial_id": plan_trial.trial_id,
            "attempt": int(attempt),
            "started_at": started_at,
        }
    )
    record = {
        "schema_version": 1,
        "record_type": "trial",
        "trial_id": plan_trial.trial_id,
        "attempt_id": attempt_id,
        "attempt": int(attempt),
        "case_id": case.case_id,
        "display_name": case.display_name,
        "solver_case": case.solver_case,
        "solver_type": (case.config.get("type") if isinstance(case.config, dict) else None),
        "graph_case": case.graph_case,
        "graph_index": case.graph_index,
        "solver_index": case.solver_index,
        "role": plan_trial.role,
        "seed": int(plan_trial.seed),
        "repetition_id": int(plan_trial.repetition_id),
        "status": status,
        "termination_reason": reason,
        "solution": solution,
        "cut_value": float(cut) if cut is not None else None,
        "solution_energy": float(-cut) if cut is not None else None,
        "config": to_json_data(config),
        "requested_num_treads": requested,
        "effective_num_treads": effective,
        "capability_limited": requested != effective,
        "capability_reason": capability_reason,
        "graph": graph.snapshot(),
        "graph_content_id": graph_content_id(graph),
        "graph_instance_id": graph_instance_id(graph),
        "node_order": list(graph.nodes),
        "timing": {
            "wall_ns": int(wall_ns),
            "wall_seconds": float(wall_ns) / 1e9,
            "scope": "solve_only" if scope == "solve_only" else "end_to_end",
            "solver_compute_seconds": solver_compute_seconds,
            "started_at": started_at,
            "finished_at": utc_now(),
        },
        "resources": resources,
        "work": work,
        "measurement_protocol": deepcopy(protocol),
        "environment_id": None,
        "reuse": False,
    }
    return record
