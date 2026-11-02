"""Benchmark specifications, deterministic expansion, and dry-run plans."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
import json
import math
from pathlib import Path

from .._version import __version__
from .experiment_ids import (
    graph_content_id,
    graph_instance_id,
    measurement_protocol_id,
    stable_hash,
    node_order,
)
from .graph import WeightedGraph
from .records import to_json_data

__all__ = [
    "PlanError",
    "BenchmarkSpec",
    "ExperimentPlan",
    "PlannedCase",
    "PlannedTrial",
    "ReferenceJob",
    "load_benchmark_spec",
    "plan_experiments",
]


class PlanError(ValueError):
    """Report a plan that cannot be expanded without ambiguity."""


_TOP_LEVEL = {
    "schema_version",
    "plan_seed",
    "trial_seeds",
    "graph_cases",
    "solver_templates",
    "solver_cases",
    "references",
    "execution",
    "measurement",
    "series_limits",
    "metrics",
    "failure_policy",
    "collect",
    "save",
    "planner_limits",
    "profile",
}

_GRAPH_CASE = {"name", "recipe", "graph", "snapshot", "grid", "multiplicity"}
_SOLVER_CASE = {"name", "template", "config", "overrides", "grid", "role"}
_AXIS = {"path", "values"}


def _mapping(value, name):
    if not isinstance(value, dict):
        raise PlanError(f"{name} must be a mapping.")
    return value


def _strict_keys(value, allowed, name):
    extra = set(value) - allowed
    if extra:
        raise PlanError(f"Unknown {name} fields: {sorted(extra)}.")


def _path(value, name="path"):
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(part, bool) or not isinstance(part, (str, int)) for part in value)
    ):
        raise PlanError(f"{name} must be a nonempty list of mapping keys or indexes.")
    return tuple(value)


_MISSING = object()


def _walk_path(data, path, *, create=False, value=_MISSING):
    current = data
    for index, part in enumerate(path[:-1]):
        if isinstance(current, dict):
            if part not in current:
                if create:
                    current[part] = {}
                else:
                    raise PlanError(f"Grid path does not exist: {path!r}.")
            current = current[part]
        elif isinstance(current, list):
            if not isinstance(part, int) or not 0 <= part < len(current):
                raise PlanError(f"Grid path does not exist: {path!r}.")
            current = current[part]
        else:
            raise PlanError(f"Grid path crosses a scalar: {path!r}.")
    last = path[-1]
    if isinstance(current, dict):
        if last not in current and not create:
            raise PlanError(f"Grid path does not exist: {path!r}.")
        if value is not _MISSING or create:
            current[last] = deepcopy(None if value is _MISSING else value)
        return current[last]
    if isinstance(current, list):
        if not isinstance(last, int) or not 0 <= last < len(current):
            raise PlanError(f"Grid path does not exist: {path!r}.")
        if value is not _MISSING or create:
            current[last] = deepcopy(None if value is _MISSING else value)
        return current[last]
    raise PlanError(f"Grid path crosses a scalar: {path!r}.")


def _axes(value, name):
    if value is None:
        return [], 1
    if not isinstance(value, list):
        raise PlanError(f"{name} must be a list of axes.")
    result = []
    seen = []
    total = 1
    for index, axis in enumerate(value):
        axis = _mapping(axis, f"{name}[{index}]")
        _strict_keys(axis, _AXIS, f"{name}[{index}]")
        path = _path(axis.get("path"), f"{name}[{index}].path")
        values = axis.get("values")
        if not isinstance(values, list) or not values:
            raise PlanError(f"{name}[{index}].values must be a nonempty list.")
        for previous in seen:
            if path == previous:
                raise PlanError(f"Duplicate grid axis path: {path!r}.")
            shorter, longer = (path, previous) if len(path) <= len(previous) else (previous, path)
            if len(shorter) < len(longer) and longer[: len(shorter)] == shorter:
                raise PlanError(f"Overlapping grid axis paths: {previous!r} and {path!r}.")
        seen.append(path)
        total *= len(values)
        result.append((path, values))
    return result, total


def _strip_scientific(value):
    """Remove lifecycle/output fields from a config for scientific identity."""
    if isinstance(value, dict):
        return {
            str(key): _strip_scientific(item)
            for key, item in value.items()
            if key not in {"seed", "num_treads", "output", "save", "checkpoint", "report", "logs"}
        }
    if isinstance(value, list):
        return [_strip_scientific(item) for item in value]
    return deepcopy(value)


def _resolved_measurement(data):
    measurement = deepcopy(data.get("measurement", {}))
    scope = measurement.get("timing_scope", "end_to_end")
    if scope == "benchmark":
        scope = "end_to_end"
    if scope not in ("end_to_end", "solve_only", "serialization/export"):
        raise PlanError("measurement.timing_scope must be end_to_end, solve_only, serialization/export, or benchmark.")
    if measurement.get("solver_lifecycle", "fresh") not in ("fresh", "shared"):
        raise PlanError("measurement.solver_lifecycle must be fresh or shared.")
    if type(measurement.get("synchronize_gpu", True)) is not bool:
        raise PlanError("measurement.synchronize_gpu must be boolean.")
    trial_output = measurement.get("trial_output", {"report": 0, "logs": 0})
    if not isinstance(trial_output, dict) or set(trial_output) - {"report", "logs"}:
        raise PlanError("measurement.trial_output must contain report and logs.")
    for key in ("report", "logs"):
        val = trial_output.get(key, 0)
        if type(val) is not int or val not in (0, 1, 2):
            raise PlanError("measurement.trial_output values must be 0, 1, or 2.")
    if "trial_save" in measurement and type(measurement["trial_save"]) is not bool:
        raise PlanError("measurement.trial_save must be boolean.")
    return {
        "timing_scope": scope,
        "declared_timing_scope": measurement.get("timing_scope", "end_to_end"),
        "solver_lifecycle": measurement.get("solver_lifecycle", "fresh"),
        "synchronize_gpu": measurement.get("synchronize_gpu", True),
        "trial_output": {"report": int(trial_output.get("report", 0)), "logs": int(trial_output.get("logs", 0))},
        "trial_save": bool(measurement.get("trial_save", False)),
    }


def _resolved_execution(data):
    execution = deepcopy(data.get("execution", {}))
    allowed = {"mode", "cpu_slots", "solver_num_treads", "gpu_policy", "backend", "max_concurrent_trials", "memory_gb"}
    extra = set(execution) - allowed
    if extra:
        raise PlanError(f"Unknown execution fields: {sorted(extra)}.")
    mode = execution.get("mode", "isolated")
    if mode not in ("isolated", "concurrent_batch"):
        raise PlanError("execution.mode must be isolated or concurrent_batch.")
    threads = execution.get("solver_num_treads", 1)
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise PlanError("execution.solver_num_treads must be a positive integer.")
    cpu_slots = execution.get("cpu_slots", 1)
    if isinstance(cpu_slots, bool) or not isinstance(cpu_slots, int) or cpu_slots < 1:
        raise PlanError("execution.cpu_slots must be a positive integer.")
    max_concurrent = execution.get("max_concurrent_trials", 1)
    if isinstance(max_concurrent, bool) or not isinstance(max_concurrent, int) or max_concurrent < 1:
        raise PlanError("execution.max_concurrent_trials must be a positive integer.")
    backend = execution.get("backend", "process")
    if backend not in ("process", "threads", "sequential"):
        raise PlanError("execution.backend must be process, threads, or sequential.")
    gpu_policy = execution.get("gpu_policy", "exclusive_one_owner")
    if not isinstance(gpu_policy, str):
        raise PlanError("execution.gpu_policy must be a string.")
    memory = execution.get("memory_gb", None)
    if memory is not None and (not isinstance(memory, (int, float)) or not math.isfinite(memory) or memory <= 0):
        raise PlanError("execution.memory_gb must be finite and positive.")
    return {
        "mode": mode,
        "cpu_slots": int(cpu_slots),
        "solver_num_treads": int(threads),
        "gpu_policy": gpu_policy,
        "backend": backend,
        "max_concurrent_trials": int(max_concurrent),
        "memory_gb": memory,
    }


def _resolved_limits(data):
    limits = deepcopy(data.get("series_limits", {}))
    extra = set(limits) - {"max_runs_total", "shots_total"}
    if extra:
        raise PlanError(f"Unknown series_limits fields: {sorted(extra)}.")
    for key in ("max_runs_total", "shots_total"):
        limits.setdefault(key, None)
        value = limits[key]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise PlanError(f"series_limits.{key} must be a nonnegative integer or null.")
    return limits


def _resolve_trial_seeds(data):
    values = data.get("trial_seeds", [0])
    if not isinstance(values, list) or not values:
        raise PlanError("trial_seeds must be a nonempty list.")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PlanError("Every trial seed must be a nonnegative integer.")
        if value in result:
            raise PlanError("trial_seeds must be unique.")
        result.append(int(value))
    return result


class PlannedCase:
    """One graph instance and one validated scientific solver config."""

    def __init__(self, *, graph, graph_case, graph_index, solver_case, solver_index, config, role="scientific"):
        self.graph = graph
        self.graph_case = str(graph_case)
        self.graph_index = int(graph_index)
        self.solver_case = str(solver_case)
        self.solver_index = int(solver_index)
        self.config = deepcopy(config)
        self.role = str(role)
        self.graph_content_id = graph_content_id(graph)
        self.graph_instance_id = graph_instance_id(graph)
        self.node_order = node_order(graph)
        self.case_id = stable_hash(
            {
                "schema": "case-v1",
                "graph_instance_id": self.graph_instance_id,
                "solver": _strip_scientific(self.config),
                "role": self.role,
            }
        )
        self.display_name = f"{self.graph_case}:{self.solver_case}:{self.graph_index}"

    def to_dict(self):
        return {
            "case_id": self.case_id,
            "display_name": self.display_name,
            "graph_case": self.graph_case,
            "graph_index": self.graph_index,
            "solver_case": self.solver_case,
            "solver_index": self.solver_index,
            "role": self.role,
            "graph_content_id": self.graph_content_id,
            "graph_instance_id": self.graph_instance_id,
            "node_order": list(self.node_order),
            "node_count": len(self.graph),
            "edge_count": len(self.graph.edges),
            "graph": self.graph.snapshot(),
            "config": to_json_data(self.config),
        }


class PlannedTrial:
    """One planned independent measurement."""

    def __init__(self, case, *, seed, num_treads, protocol, repetition_id=0, role="scientific"):
        self.case = case
        self.seed = int(seed)
        self.num_treads = int(num_treads)
        self.protocol = deepcopy(protocol)
        self.repetition_id = int(repetition_id)
        self.role = str(role)
        self.trial_id = stable_hash(
            {
                "schema": "trial-v1",
                "case_id": case.case_id,
                "seed": self.seed,
                "num_treads": self.num_treads,
                "repetition_id": self.repetition_id,
                "role": self.role,
                "measurement_protocol": measurement_protocol_id(protocol),
                "implementation_version": __version__,
            }
        )

    def to_dict(self):
        return {
            "trial_id": self.trial_id,
            "case_id": self.case.case_id,
            "display_name": self.case.display_name,
            "seed": self.seed,
            "num_treads": self.num_treads,
            "repetition_id": self.repetition_id,
            "role": self.role,
            "measurement_protocol": deepcopy(self.protocol),
            "config": to_json_data(self.case.config),
            "graph_instance_id": self.case.graph_instance_id,
            "graph_content_id": self.case.graph_content_id,
        }


class ReferenceJob:
    """Describe one reference computation without running it."""

    def __init__(self, *, graph, kind, config, seed, include_minimum=False, source=None):
        self.graph = graph.copy()
        self.kind = str(kind)
        self.config = deepcopy(config)
        self.seed = seed
        self.include_minimum = bool(include_minimum)
        self.source = deepcopy(source)
        self.graph_content_id = graph_content_id(graph)
        self.graph_instance_id = graph_instance_id(graph)
        self.reference_id = stable_hash(
            {
                "schema": "reference-v1",
                "graph_instance_id": self.graph_instance_id,
                "kind": self.kind,
                "config": _strip_scientific(self.config),
                "seed": self.seed,
                "include_minimum": self.include_minimum,
                "implementation_version": __version__,
            }
        )

    def to_dict(self):
        return {
            "reference_id": self.reference_id,
            "kind": self.kind,
            "role": "reference",
            "graph_content_id": self.graph_content_id,
            "graph_instance_id": self.graph_instance_id,
            "node_order": list(self.graph.nodes),
            "config": to_json_data(self.config),
            "seed": self.seed,
            "include_minimum": self.include_minimum,
            "source": to_json_data(self.source),
        }


class ExperimentPlan:
    """Resolved graph, solver, trial, and reference identities."""

    def __init__(self, *, spec, cases, trials, rejected, reference_jobs, graphs, raw_solver_configs, profile=False):
        self.spec = deepcopy(spec)
        self.cases = list(cases)
        self.trials = list(trials)
        self.rejected = list(rejected)
        self.reference_jobs = list(reference_jobs)
        self.graphs = list(graphs)
        self.raw_solver_configs = int(raw_solver_configs)
        self.profile = bool(profile)
        self.counts = {
            "graph_instances": len(self.graphs),
            "cases": len(self.cases),
            "raw_solver_configs": self.raw_solver_configs,
            "accepted_solver_configs": len(self.cases),
            "rejected_solver_configs": len(self.rejected),
            "trials": len(self.trials),
            "reference_jobs": len(self.reference_jobs),
            "planned_top_level_runs": len(self.trials) + len(self.reference_jobs),
        }
        self.summary = (
            f"graphs={len(self.graphs)} cases={len(self.cases)} "
            f"trials={len(self.trials)} rejected={len(self.rejected)} "
            f"references={len(self.reference_jobs)}"
        )

    @property
    def result(self):
        return to_json_data(
            {
                "schema_version": 1,
                "record_type": "experiment_plan",
                "package_version": __version__,
                "profile": self.profile,
                "summary": self.summary,
                "counts": deepcopy(self.counts),
                "seeds": deepcopy(self.spec.get("trial_seeds", [])),
                "execution": deepcopy(self.spec.get("execution", {})),
                "measurement": deepcopy(self.spec.get("measurement", {})),
                "series_limits": deepcopy(self.spec.get("series_limits", {})),
                "graph_instances": [graph.snapshot() for graph in self.graphs],
                "cases": [case.to_dict() for case in self.cases],
                "trials": [trial.to_dict() for trial in self.trials],
                "reference_jobs": [job.to_dict() for job in self.reference_jobs],
                "rejected": deepcopy(self.rejected),
            }
        )

    @property
    def plan_hash(self):
        return stable_hash({"schema": "plan-v1", "plan": self.result})

    def to_dict(self):
        return self.result


def _make_graph(data):
    if isinstance(data, WeightedGraph):
        return data.copy(), deepcopy(data.snapshot().get("recipe"))
    if not isinstance(data, dict):
        raise PlanError("Graph case needs recipe, graph, or snapshot.")
    if "recipe" in data:
        from ..generators.graphs import GraphFactory

        recipe = deepcopy(data["recipe"])
        return GraphFactory.from_recipe(recipe), recipe
    if "nodes" in data and "edges" in data:
        return WeightedGraph.from_snapshot(data), deepcopy(data.get("recipe"))
    if "snapshot" in data:
        return WeightedGraph.from_snapshot(data["snapshot"]), deepcopy(data["snapshot"].get("recipe"))
    raise PlanError("Graph case needs recipe, graph, or snapshot.")


def _graph_variants(data, limits):
    case = _mapping(data, "graph case")
    _strict_keys(case, _GRAPH_CASE, "graph case")
    if not case.get("name"):
        raise PlanError("A graph case needs a nonempty name.")
    sources = [key for key in ("recipe", "graph", "snapshot") if key in case]
    if len(sources) != 1:
        raise PlanError("A graph case needs exactly one of recipe, graph, snapshot.")
    if "multiplicity" in case and case["multiplicity"] != 1:
        raise PlanError("Explicit graph multiplicity is not implemented; provide distinct graph cases.")
    source = deepcopy(case[sources[0]])
    axes, total = _axes(case.get("grid"), f"graph case {case['name']}.grid")
    if total > limits["max_cases"]:
        raise PlanError("Graph grid exceeds planner_limits.max_cases.")
    variants = []
    for index, values in enumerate(product(*(axis[1] for axis in axes)) if axes else [()]):
        item = deepcopy(source)
        for (path, _), value in zip(axes, values):
            _walk_path(item, path, create=False, value=value)
        if "recipe" in case:
            item = {"recipe": item}
        elif "graph" in case:
            item = {"graph": item}
        else:
            item = {"snapshot": item}
        graph, recipe = _make_graph(item)
        variants.append((index, graph))
    return variants


def _apply_override(config, override):
    override = _mapping(override, "solver override")
    _strict_keys(override, {"path", "value"}, "solver override")
    path = _path(override.get("path"), "override.path")
    _walk_path(config, path, create=False, value=override.get("value"))


def _solver_variants(data, templates, graph, limits):
    case = _mapping(data, "solver case")
    _strict_keys(case, _SOLVER_CASE, "solver case")
    name = case.get("name")
    if not name or not isinstance(name, str):
        raise PlanError("A solver case needs a nonempty name.")
    if ("template" in case) == ("config" in case):
        raise PlanError(f"Solver case {name!r} needs exactly one of template or config.")
    if "template" in case:
        template = case["template"]
        if not isinstance(template, str) or template not in templates:
            raise PlanError(f"Unknown solver template: {template!r}.")
        base = deepcopy(templates[template])
    else:
        base = deepcopy(case["config"])
    for override in case.get("overrides", []):
        _apply_override(base, override)
    axes, total = _axes(case.get("grid"), f"solver case {name}.grid")
    if total > limits["max_cases"]:
        raise PlanError("Solver grid exceeds planner_limits.max_cases.")
    variants = []
    for index, values in enumerate(product(*(axis[1] for axis in axes)) if axes else [()]):
        config = deepcopy(base)
        for (path, _), value in zip(axes, values):
            _walk_path(config, path, create=False, value=value)
        config.setdefault("schema_version", 1)
        if "type" not in config:
            raise PlanError(f"Solver case {name!r} has no type.")
        variants.append((index, config))
    return name, variants


def _solver_available(config):
    """Return a concrete capability error for a GPU solver when unavailable."""
    text = json.dumps(config, ensure_ascii=False, sort_keys=True).lower()
    if '"gpu' not in text and "gpu_" not in text:
        return None
    try:
        import importlib.util

        if importlib.util.find_spec("cupy") is None:
            return "CuPy is not installed."
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            return "No CUDA device is visible."
    except Exception as error:  # dependency/device-specific
        return f"GPU capability check failed: {type(error).__name__}: {error}"
    return None


def _validate_config(config, graph):
    from .factory import SolverFactory

    unavailable = _solver_available(config)
    if unavailable:
        raise PlanError(unavailable)
    solver = SolverFactory.from_config(config)
    probe = deepcopy(config)
    # The solver's validation is the authoritative closed schema.
    solver._validate(probe, graph)
    return solver.algorithm_type


def _reference_policy(data, graph):
    from .references import ReferencePolicy

    return ReferencePolicy(data).jobs_for(graph)


class BenchmarkSpec:
    """Validated benchmark-spec wrapper and deterministic planner."""

    def __init__(self, data):
        if not isinstance(data, dict):
            raise PlanError("A benchmark specification must be a mapping.")
        self.data = deepcopy(data)
        _strict_keys(self.data, _TOP_LEVEL, "benchmark specification")
        if self.data.get("schema_version", 1) != 1:
            raise PlanError("Unsupported benchmark schema_version.")
        if "graph_cases" not in self.data or not self.data["graph_cases"]:
            raise PlanError("graph_cases must be a nonempty list.")
        if "solver_cases" not in self.data or not self.data["solver_cases"]:
            raise PlanError("solver_cases must be a nonempty list.")
        templates = self.data.get("solver_templates", {})
        if not isinstance(templates, dict):
            raise PlanError("solver_templates must be a mapping.")
        for name, template in templates.items():
            if not isinstance(name, str) or not isinstance(template, dict):
                raise PlanError("Every solver template must be a named mapping.")
        self.trial_seeds = _resolve_trial_seeds(self.data)
        self.execution = _resolved_execution(self.data)
        self.measurement = _resolved_measurement(self.data)
        self.series_limits = _resolved_limits(self.data)
        planner = deepcopy(self.data.get("planner_limits", {}))
        if not isinstance(planner, dict) or set(planner) - {"max_cases", "max_trials"}:
            raise PlanError("planner_limits may contain max_cases and max_trials.")
        planner.setdefault("max_cases", 50000)
        planner.setdefault("max_trials", 200000)
        for key, val in planner.items():
            if isinstance(val, bool) or not isinstance(val, int) or val < 1:
                raise PlanError(f"planner_limits.{key} must be a positive integer.")
        self.planner_limits = planner
        metrics = self.data.get("metrics", {})
        if not isinstance(metrics, dict) or set(metrics) - {"fields", "success"}:
            raise PlanError("metrics must contain only fields and success.")
        if "fields" in metrics and (
            not isinstance(metrics["fields"], list)
            or any(not isinstance(name, str) for name in metrics["fields"])
        ):
            raise PlanError("metrics.fields must be a list of names.")
        if "success" in metrics:
            success = metrics["success"]
            if not isinstance(success, dict) or success.get("type") not in (
                "fraction_of_exact_max",
                "absolute_cut",
            ):
                raise PlanError("metrics.success has an unknown type.")
        failure = self.data.get("failure_policy", {})
        if not isinstance(failure, dict) or set(failure) - {"on_trial_error", "retry_failed"}:
            raise PlanError("Unknown failure_policy field.")
        if failure.get("on_trial_error", "continue") not in ("continue", "fail_fast"):
            raise PlanError("failure_policy.on_trial_error must be continue or fail_fast.")
        if "retry_failed" in failure and type(failure["retry_failed"]) is not bool:
            raise PlanError("failure_policy.retry_failed must be boolean.")
        collect = self.data.get("collect", {})
        if not isinstance(collect, dict) or set(collect) - {"trial_records", "raw_measurements"}:
            raise PlanError("Unknown collect field.")
        save = self.data.get("save")
        if save is not None:
            if not isinstance(save, dict) or "path" not in save or "filename" not in save:
                raise PlanError("save needs path and filename.")
            if set(save) - {
                "path",
                "filename",
                "makedir",
                "savedir",
                "atts",
                "save_original_data",
                "format",
                "DPI",
                "metadata",
            }:
                raise PlanError("Unknown save field.")

    @property
    def measurement_protocol(self):
        protocol = deepcopy(self.measurement)
        protocol["mode"] = self.execution["mode"]
        protocol["backend"] = self.execution["backend"]
        protocol["cpu_slots"] = self.execution["cpu_slots"]
        protocol["memory_gb"] = self.execution["memory_gb"]
        return protocol

    def plan(self, *, trial_seeds=None, profile=False):
        """Expand all graphs, solver branches, seeds, and references.

        Parameters
        ----------
        trial_seeds : list of int, optional
            Override the specification seeds.  The parallelism profiler uses
            one selected seed here.
        profile : bool, default False
            Mark the plan as a profile plan.  The CPU axis is added by the
            profiler, not by this planner.
        """
        seeds = self.trial_seeds if trial_seeds is None else list(trial_seeds)
        if not seeds:
            raise PlanError("A plan needs at least one trial seed.")
        graph_variants = []
        for case in self.data["graph_cases"]:
            case_name = case.get("name") if isinstance(case, dict) else None
            for index, graph in _graph_variants(case, self.planner_limits):
                graph_variants.append((case_name, index, graph))
        graphs = []
        seen_instances = {}
        for case_name, index, graph in graph_variants:
            instance = graph_instance_id(graph)
            if instance in seen_instances:
                raise PlanError(
                    "Duplicate graph instance; use a distinct graph case or explicit multiplicity."
                )
            seen_instances[instance] = graph
            graphs.append(graph)
        cases = []
        rejected = []
        raw_count = 0
        protocol = self.measurement_protocol
        for case_name, graph_index, graph in graph_variants:
            for solver_data in self.data["solver_cases"]:
                name, variants = _solver_variants(
                    solver_data, self.data.get("solver_templates", {}), graph, self.planner_limits
                )
                raw_count += len(variants)
                for solver_index, config in variants:
                    try:
                        _validate_config(config, graph)
                    except Exception as error:
                        rejected.append(
                            {
                                "graph_case": case_name,
                                "graph_index": graph_index,
                                "solver_case": name,
                                "solver_index": solver_index,
                                "config": to_json_data(config),
                                "reason": str(error),
                                "error_type": type(error).__name__,
                            }
                        )
                        continue
                    roles = solver_data.get("role", "scientific")
                    case = PlannedCase(
                        graph=graph,
                        graph_case=case_name,
                        graph_index=graph_index,
                        solver_case=name,
                        solver_index=solver_index,
                        config=config,
                        role=roles if isinstance(roles, str) else "scientific",
                    )
                    if len(cases) >= self.planner_limits["max_cases"]:
                        raise PlanError("Plan exceeds planner_limits.max_cases; reduce the grid explicitly.")
                    cases.append(case)
        trials = []
        for case in cases:
            for seed in seeds:
                trials.append(
                    PlannedTrial(
                        case,
                        seed=seed,
                        num_treads=self.execution["solver_num_treads"],
                        protocol=protocol,
                        role="scientific",
                    )
                )
        if len(trials) > self.planner_limits["max_trials"]:
            raise PlanError("Plan exceeds planner_limits.max_trials; reduce the grid explicitly.")
        reference_jobs = []
        for graph in graphs:
            reference_jobs.extend(_reference_policy(self.data.get("references", {}), graph))
        # Reference jobs are per graph instance, not per solver case.
        unique_jobs = {}
        for job in reference_jobs:
            unique_jobs.setdefault(job.reference_id, job)
        reference_jobs = list(unique_jobs.values())
        return ExperimentPlan(
            spec={
                **deepcopy(self.data),
                "trial_seeds": seeds,
                "execution": self.execution,
                "measurement": self.measurement,
                "series_limits": self.series_limits,
            },
            cases=cases,
            trials=trials,
            rejected=rejected,
            reference_jobs=reference_jobs,
            graphs=graphs,
            raw_solver_configs=raw_count,
            profile=profile,
        )


def load_benchmark_spec(path):
    """Load and validate a benchmark JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return BenchmarkSpec(data)


def plan_experiments(spec, *, trial_seeds=None):
    """Plan a benchmark from a mapping or a loaded spec."""
    if not isinstance(spec, BenchmarkSpec):
        spec = BenchmarkSpec(spec)
    return spec.plan(trial_seeds=trial_seeds)
