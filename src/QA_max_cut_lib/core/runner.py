"""Execute benchmark plans with independent trials and write-once records."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from copy import deepcopy
from datetime import datetime, timezone
import json
import multiprocessing
import os
from time import perf_counter
from types import SimpleNamespace

from .experiment_ids import stable_hash
from .experiment_spec import BenchmarkSpec, PlanError, plan_experiments
from .graph import WeightedGraph
from .measurement import (
    CampaignBudget,
    estimate_trial_shots,
    execute_trial,
    measurement_environment,
)
from .records import to_json_data
from .series_analysis import analyze_series
from .series_store import ExperimentStore
from .experiment_spec import PlannedTrial
from .._version import __version__

__all__ = ["ExperimentRunner", "ExperimentSeries", "run_series"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _case_payload(case):
    return {
        "graph": case.graph.snapshot(),
        "config": deepcopy(case.config),
        "case_id": case.case_id,
        "display_name": case.display_name,
        "solver_case": case.solver_case,
        "solver_type": case.config.get("type"),
        "graph_case": case.graph_case,
        "graph_index": case.graph_index,
        "solver_index": case.solver_index,
        "role": case.role,
        "graph_content_id": case.graph_content_id,
        "graph_instance_id": case.graph_instance_id,
        "node_order": list(case.node_order),
    }


def _trial_payload(trial):
    return {
        "trial_id": trial.trial_id,
        "seed": trial.seed,
        "num_treads": trial.num_treads,
        "repetition_id": trial.repetition_id,
        "role": trial.role,
        "protocol": deepcopy(trial.protocol),
    }


def _worker_execute(payload):
    """Top-level spawn worker: no shared solver object is transferred."""
    try:
        from threadpoolctl import threadpool_limits

        graph = WeightedGraph.from_snapshot(payload["case"]["graph"])
        case_data = payload["case"]
        case = SimpleNamespace(
            graph=graph,
            config=case_data["config"],
            case_id=case_data["case_id"],
            display_name=case_data["display_name"],
            solver_case=case_data["solver_case"],
            solver_type=case_data["solver_type"],
            graph_case=case_data["graph_case"],
            graph_index=case_data["graph_index"],
            solver_index=case_data["solver_index"],
            role=case_data["role"],
            graph_content_id=case_data["graph_content_id"],
            graph_instance_id=case_data["graph_instance_id"],
            node_order=case_data["node_order"],
        )
        trial_data = payload["trial"]
        trial = SimpleNamespace(
            case=case,
            trial_id=trial_data["trial_id"],
            seed=trial_data["seed"],
            num_treads=trial_data["num_treads"],
            repetition_id=trial_data["repetition_id"],
            role=trial_data["role"],
            protocol=trial_data["protocol"],
        )
        with threadpool_limits(limits=max(1, int(trial.num_treads))):
            return execute_trial(
                trial,
                attempt=int(payload.get("attempt", 0)),
                protocol=trial.protocol,
            )
    except Exception as error:
        return {
            "record_type": "worker_error",
            "trial_id": payload.get("trial", {}).get("trial_id"),
            "status": "failed",
            "reason": f"{type(error).__name__}: {error}",
        }


def _skipped(trial, reason):
    return {
        "schema_version": 1,
        "record_type": "trial",
        "trial_id": trial.trial_id,
        "attempt_id": stable_hash({"schema": "attempt-v1", "trial_id": trial.trial_id, "attempt": 0}),
        "attempt": 0,
        "case_id": trial.case.case_id,
        "display_name": trial.case.display_name,
        "solver_case": trial.case.solver_case,
        "solver_type": trial.case.config.get("type"),
        "graph_case": trial.case.graph_case,
        "graph_index": trial.case.graph_index,
        "solver_index": trial.case.solver_index,
        "role": trial.role,
        "seed": trial.seed,
        "repetition_id": trial.repetition_id,
        "status": "skipped",
        "termination_reason": reason,
        "solution": None,
        "cut_value": None,
        "solution_energy": None,
        "config": to_json_data(trial.case.config),
        "requested_num_treads": trial.num_treads,
        "effective_num_treads": None,
        "capability_limited": False,
        "capability_reason": None,
        "graph": trial.case.graph.snapshot(),
        "graph_content_id": trial.case.graph_content_id,
        "graph_instance_id": trial.case.graph_instance_id,
        "node_order": list(trial.case.node_order),
        "timing": {"wall_ns": None, "wall_seconds": None, "scope": trial.protocol.get("timing_scope")},
        "resources": {},
        "work": {"counters": {}},
        "measurement_protocol": deepcopy(trial.protocol),
        "environment_id": None,
        "reuse": False,
    }


class ExperimentSeries:
    """Independent records from one campaign."""

    def __init__(self, *, plan, references, trials, errors, environment, budget, wall_seconds, makespan_seconds, mode, save_status):
        self.plan = plan
        self.references = list(references)
        self.trials = list(trials)
        self.errors = list(errors)
        self.environment = environment
        self.budget = budget
        self.wall_seconds = float(wall_seconds)
        self.makespan_seconds = float(makespan_seconds)
        self.mode = mode
        self.save_status = save_status
        self.analysis = analyze_series(
            self.trials,
            self.references,
            metrics_config=plan.spec.get("metrics", {}),
            plan=plan.result,
            errors=self.errors,
        )
        self._refresh_summary()

    def _refresh_summary(self):
        analysis = self.analysis
        self.summary = (
            f"planned={analysis['planned']} valid={analysis['valid']} "
            f"failed={analysis['failed']} skipped={analysis['skipped']} "
            f"success(valid)={analysis['success_conditional_on_valid']} "
            f"wall={self.wall_seconds:.3f}s"
        )

    @property
    def result(self):
        return to_json_data(
            {
                "schema_version": 1,
                "record_type": "experiment_series",
                "package_version": __version__,
                "mode": self.mode,
                "summary": self.summary,
                "plan": self.plan.result,
                "references": self.references,
                "trials": self.trials,
                "analysis": {key: value for key, value in self.analysis.items() if key != "trials"},
                "errors": self.errors,
                "environment": self.environment,
                "budget": self.budget,
                "timing": {
                    "wall_seconds": self.wall_seconds,
                    "makespan_seconds": self.makespan_seconds,
                },
                "save": self.save_status,
            }
        )

    def load(self):
        """Return a fresh snapshot; the in-memory campaign is immutable."""
        return deepcopy(self.result)


class ExperimentRunner:
    """Run a resolved benchmark plan.

    Parameters
    ----------
    spec : BenchmarkSpec or mapping
        The benchmark to run.
    reuse_completed : bool, default False
        Reuse only complete records whose trial id, protocol, and measurement
        environment match.  A reused timing record is not a new repeat.
    """

    def __init__(self, spec, *, reuse_completed=False):
        self.spec = spec if isinstance(spec, BenchmarkSpec) else BenchmarkSpec(spec)
        self.plan = self.spec.plan()
        self.reuse_completed = bool(reuse_completed)
        self.environment = measurement_environment()
        self.store = ExperimentStore(self.spec.data.get("save"), default_filename="experiment_series")
        self.series = None
        self.errors = []
        self._existing = {}
        if self.store.enabled and self.reuse_completed:
            existing = self.store.read_trial_records()
            for trial_id, record in existing.items():
                if record.get("environment_id") == self.environment["environment_id"]:
                    self._existing[trial_id] = record

    def _admission(self):
        limits = self.spec.series_limits
        max_runs = limits["max_runs_total"]
        total = len(self.plan.trials) + len(self.plan.reference_jobs)
        if max_runs is not None and total > max_runs:
            raise PlanError(
                f"Plan needs {total} top-level runs but max_runs_total={max_runs}; "
                "reduce the grid explicitly."
            )
        if limits["shots_total"] is not None:
            total_shots = 0
            for trial in self.plan.trials:
                estimate = estimate_trial_shots(trial.case.config)
                if estimate is None:
                    raise PlanError(
                        "A finite campaign shots_total requires a finite per-trial shot cap."
                    )
                total_shots += int(estimate)
            if total_shots > limits["shots_total"]:
                raise PlanError(
                    f"Plan reserves {total_shots} shots but shots_total={limits['shots_total']}; "
                    "reduce the grid explicitly."
                )
        if self.spec.execution["mode"] == "isolated" and self.spec.execution["solver_num_treads"] > self.spec.execution["cpu_slots"]:
            raise PlanError("solver_num_treads exceeds execution.cpu_slots.")

    def _attach(self, record, trial):
        record = deepcopy(record)
        record["environment_id"] = record.get("environment_id") or self.environment["environment_id"]
        record["measurement_protocol"] = deepcopy(trial.protocol)
        record["save"] = self.store.snapshot()
        return record

    def _run_one(self, trial, *, budget):
        if self.reuse_completed and trial.trial_id in self._existing:
            record = deepcopy(self._existing[trial.trial_id])
            record["reuse"] = True
            return record
        if trial.num_treads > self.spec.execution["cpu_slots"]:
            return _skipped(trial, "requested_num_treads_exceeds_cpu_slots")
        if not budget.reserve_run(1):
            return _skipped(trial, "max_runs_total")
        estimate = estimate_trial_shots(trial.case.config)
        if not budget.reserve_shots(estimate):
            budget.release_run(1)
            return _skipped(trial, "shots_total_reservation")
        attempts = []
        record = execute_trial(trial, attempt=0)
        attempts.append(
            {
                "attempt": 0,
                "attempt_id": record.get("attempt_id"),
                "status": record.get("status"),
                "termination_reason": record.get("termination_reason"),
            }
        )
        used = record.get("resources", {}).get("shots_used", 0)
        budget.settle_shots(estimate, used)
        budget.runs_completed += 1
        if record.get("status") == "failed" and self.spec.data.get("failure_policy", {}).get("retry_failed", False):
            second = execute_trial(trial, attempt=1)
            attempts.append(
                {
                    "attempt": 1,
                    "attempt_id": second.get("attempt_id"),
                    "status": second.get("status"),
                    "termination_reason": second.get("termination_reason"),
                }
            )
            second["attempt_history"] = attempts
            record = second
            budget.runs_completed += 1
        record["attempt_history"] = attempts
        record = self._attach(record, trial)
        if self.store.enabled:
            self.store.write_record(record)
        return record

    def _run_references(self, budget):
        from .references import ReferencePolicy

        policy = ReferencePolicy(self.spec.data.get("references", {}))
        records = []
        for job in self.plan.reference_jobs:
            if not budget.reserve_run(1):
                records.append(
                    {
                        "reference_id": job.reference_id,
                        "status": "skipped",
                        "kind": job.kind,
                        "reason": "max_runs_total",
                        "graph_content_id": job.graph_content_id,
                        "graph_instance_id": job.graph_instance_id,
                    }
                )
                continue
            record = policy.run_job(job)
            budget.runs_completed += 1
            record = self._attach(record, SimpleNamespace(protocol=self.spec.measurement_protocol))
            records.append(record)
            if self.store.enabled:
                self.store.write_record(record)
        return records

    def _run_sequential(self, budget, references):
        trials = []
        fail_fast = self.spec.data.get("failure_policy", {}).get("on_trial_error") == "fail_fast"
        for trial in self.plan.trials:
            record = self._run_one(trial, budget=budget)
            trials.append(record)
            if record.get("status") == "failed":
                self.errors.append({"trial_id": trial.trial_id, "reason": record.get("termination_reason")})
                if fail_fast:
                    break
        return trials

    def _run_concurrent(self, budget, references):
        max_workers = min(
            self.spec.execution["max_concurrent_trials"],
            max(1, self.spec.execution["cpu_slots"] // max(1, self.spec.execution["solver_num_treads"])),
        )
        if max_workers < 1:
            raise PlanError("Concurrent execution has no CPU capacity.")
        backend = self.spec.execution["backend"]
        Executor = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
        common = {}
        if Executor is ProcessPoolExecutor:
            common["mp_context"] = multiprocessing.get_context("spawn")
        trials = []
        fail_fast = self.spec.data.get("failure_policy", {}).get("on_trial_error") == "fail_fast"
        iterator = iter(self.plan.trials)
        pending = {}
        submitted = 0
        with Executor(max_workers=max_workers, **common) as pool:
            while True:
                while len(pending) < max_workers:
                    try:
                        trial = next(iterator)
                    except StopIteration:
                        break
                    if trial.num_treads > self.spec.execution["cpu_slots"]:
                        trials.append(_skipped(trial, "requested_num_treads_exceeds_cpu_slots"))
                        continue
                    if not budget.reserve_run(1):
                        trials.append(_skipped(trial, "max_runs_total"))
                        continue
                    estimate = estimate_trial_shots(trial.case.config)
                    if not budget.reserve_shots(estimate):
                        budget.release_run(1)
                        trials.append(_skipped(trial, "shots_total_reservation"))
                        continue
                    payload = {
                        "case": _case_payload(trial.case),
                        "trial": _trial_payload(trial),
                        "attempt": 0,
                    }
                    future = pool.submit(_worker_execute, payload)
                    pending[future] = (trial, estimate)
                    submitted += 1
                if not pending:
                    break
                done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    trial, estimate = pending.pop(future)
                    result = future.result()
                    if result.get("record_type") == "worker_error":
                        record = _skipped(trial, f"worker_error:{result.get('reason')}")
                        record["attempt_id"] = stable_hash(
                            {"schema": "attempt-v1", "trial_id": trial.trial_id, "attempt": 0}
                        )
                        budget.settle_shots(estimate, 0)
                        budget.runs_completed += 1
                    else:
                        record = result
                        used = record.get("resources", {}).get("shots_used", 0)
                        budget.settle_shots(estimate, used)
                        budget.runs_completed += 1
                    record["attempt_history"] = [
                        {
                            "attempt": 0,
                            "attempt_id": record.get("attempt_id"),
                            "status": record.get("status"),
                            "termination_reason": record.get("termination_reason"),
                        }
                    ]
                    record = self._attach(record, trial)
                    trials.append(record)
                    if self.store.enabled:
                        self.store.write_record(record)
                    if record.get("status") == "failed":
                        self.errors.append({"trial_id": trial.trial_id, "reason": record.get("termination_reason")})
                        if fail_fast:
                            for pending_future in list(pending):
                                pending_future.cancel()
                            iterator = iter(())
        # Stable report order, independent of completion order.
        order = {trial.trial_id: index for index, trial in enumerate(self.plan.trials)}
        trials.sort(key=lambda item: order.get(item.get("trial_id"), 10**12))
        return trials

    def run(self):
        """Run the campaign and return an independent series snapshot."""
        self._admission()
        self.errors = []
        budget = CampaignBudget(
            max_runs=self.spec.series_limits["max_runs_total"],
            shots_total=self.spec.series_limits["shots_total"],
        )
        start = perf_counter()
        references = self._run_references(budget)
        for reference in references:
            if reference.get("status") == "failed":
                self.errors.append({"reference_id": reference.get("reference_id"), "reason": reference.get("reason")})
        if self.spec.execution["mode"] == "isolated":
            trials = self._run_sequential(budget, references)
            makespan = perf_counter() - start
        else:
            trials = self._run_concurrent(budget, references)
            makespan = perf_counter() - start
        wall = perf_counter() - start
        self.series = ExperimentSeries(
            plan=self.plan,
            references=references,
            trials=trials,
            errors=self.errors,
            environment=self.environment,
            budget=budget.snapshot(),
            wall_seconds=wall,
            makespan_seconds=makespan,
            mode=self.spec.execution["mode"],
            save_status=self.store.snapshot(),
        )
        if self.store.enabled:
            self.store.write_manifest(self.series.load())
        return self.series


def run_series(spec, *, reuse_completed=False):
    """Convenience wrapper for ``ExperimentRunner(spec).run()``."""
    return ExperimentRunner(spec, reuse_completed=reuse_completed).run()
