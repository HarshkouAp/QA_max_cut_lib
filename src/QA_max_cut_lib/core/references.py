"""Reference policies for exact, GW, and best-known cut values."""

from __future__ import annotations

from copy import deepcopy
import math
from time import perf_counter

from .experiment_ids import graph_content_id, graph_instance_id, stable_hash
from .graph import WeightedGraph
from .records import to_json_data

__all__ = ["ReferencePolicy", "ReferenceJob"]


class ReferenceJob:
    """Describe one reference computation.

    Parameters
    ----------
    graph : WeightedGraph
        Input graph for the reference.
    kind : str
        ``exact_max_cut``, ``exact_min_cut``, ``gw_rounded_cut``, or
        ``best_known_cut``.
    config : dict
        Resolved solver or source configuration.
    seed : int or None
        Reference-specific seed.
    source : dict or None
        Explicit provenance for a best-known value.
    """

    def __init__(self, *, graph, kind, config, seed, source=None):
        self.graph = graph.copy()
        self.kind = str(kind)
        self.config = deepcopy(config)
        self.seed = seed
        self.source = deepcopy(source)
        self.graph_content_id = graph_content_id(graph)
        self.graph_instance_id = graph_instance_id(graph)
        self.reference_id = stable_hash(
            {
                "schema": "reference-v1",
                "graph_instance_id": self.graph_instance_id,
                "kind": self.kind,
                "config": self.config,
                "seed": self.seed,
                "source": self.source,
            }
        )

    def to_dict(self):
        return {
            "reference_id": self.reference_id,
            "kind": self.kind,
            "graph_content_id": self.graph_content_id,
            "graph_instance_id": self.graph_instance_id,
            "node_order": list(self.graph.nodes),
            "config": to_json_data(self.config),
            "seed": self.seed,
            "source": to_json_data(self.source),
        }


class ReferencePolicy:
    """Select and execute the configured reference tier.

    Parameters
    ----------
    config : dict
        A tiered policy with optional ``exact``, ``gw``, and ``best_known``
        sections.  Thresholds are explicit user parameters; this class does
        not assume universal 20/100/1000 vertex limits.

    Attributes
    ----------
    last_references : list of dict
        Independent records from the most recent execution.
    """

    def __init__(self, config=None):
        config = deepcopy(config or {})
        if set(config) - {"type", "exact", "gw", "best_known"}:
            raise ValueError("Unknown reference-policy field.")
        config.setdefault("type", "tiered")
        if config["type"] not in ("tiered", "exact", "gw", "best_known", "none"):
            raise ValueError("Unknown reference-policy type.")
        exact = deepcopy(config.get("exact", {}))
        gw = deepcopy(config.get("gw", {}))
        known = deepcopy(config.get("best_known", {}))
        if set(exact) - {"max_vertices", "include_minimum", "config"}:
            raise ValueError("Unknown exact-reference field.")
        exact.setdefault("max_vertices", 0)
        exact.setdefault("include_minimum", False)
        exact.setdefault("config", {"type": "exact_cpu", "parameters": {}})
        if (
            isinstance(exact["max_vertices"], bool)
            or not isinstance(exact["max_vertices"], int)
            or exact["max_vertices"] < 0
            or exact["max_vertices"] > 30
        ):
            raise ValueError("exact.max_vertices must be an integer in [0, 30].")
        if type(exact["include_minimum"]) is not bool:
            raise ValueError("exact.include_minimum must be boolean.")
        if not isinstance(exact["config"], dict):
            raise ValueError("exact.config must be a mapping.")
        exact["config"].setdefault("type", "exact_cpu")
        exact["config"].setdefault("parameters", {})
        if set(gw) - {"max_vertices", "config", "seed"}:
            raise ValueError("Unknown GW-reference field.")
        gw.setdefault("max_vertices", 0)
        if (
            isinstance(gw["max_vertices"], bool)
            or not isinstance(gw["max_vertices"], int)
            or gw["max_vertices"] < 0
        ):
            raise ValueError("gw.max_vertices must be a nonnegative integer.")
        gw.setdefault("config", {"type": "gw_cpu", "parameters": {"num_hyperplanes": 10}})
        if not isinstance(gw["config"], dict):
            raise ValueError("gw.config must be a mapping.")
        gw.setdefault("seed", None)
        if set(known) - {"sources"}:
            raise ValueError("Unknown best-known-reference field.")
        sources = known.setdefault("sources", [])
        if not isinstance(sources, list):
            raise ValueError("best_known.sources must be a list.")
        for source in sources:
            if not isinstance(source, dict) or {"graph_content_id", "value"} - set(source):
                raise ValueError("Each best-known source needs graph_content_id and value.")
            value = source["value"]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("A best-known value must be finite.")
        self._config = {
            "type": config["type"],
            "exact": exact,
            "gw": gw,
            "best_known": {"sources": sources},
        }
        self.last_references = []

    @property
    def config(self):
        return deepcopy(self._config)

    def jobs_for(self, graph):
        """Return the planned reference jobs for one graph."""
        if not isinstance(graph, WeightedGraph):
            raise TypeError("graph must be a WeightedGraph.")
        policy = self._config
        n = len(graph)
        jobs = []
        exact = policy["exact"]
        gw = policy["gw"]
        known = policy["best_known"]
        if policy["type"] in ("tiered", "exact") and n <= exact["max_vertices"]:
            config = deepcopy(exact["config"])
            config["parameters"]["max_vertices"] = exact["max_vertices"]
            config.pop("save", None)
            config.pop("output", None)
            jobs.append(
                ReferenceJob(
                    graph=graph,
                    kind="exact_max_cut",
                    config=config,
                    seed=None,
                )
            )
            if exact["include_minimum"]:
                jobs.append(
                    ReferenceJob(
                        graph=graph,
                        kind="exact_min_cut",
                        config=config,
                        seed=None,
                    )
                )
            return jobs
        if policy["type"] in ("tiered", "gw") and n <= gw["max_vertices"]:
            config = deepcopy(gw["config"])
            config.setdefault("type", "gw_cpu")
            config.pop("save", None)
            config.pop("output", None)
            return [
                ReferenceJob(
                    graph=graph,
                    kind="gw_rounded_cut",
                    config=config,
                    seed=gw["seed"],
                )
            ]
        matches = [
            item
            for item in known["sources"]
            if item.get("graph_content_id") == graph_content_id(graph)
        ]
        if matches:
            source = deepcopy(matches[0])
            return [
                ReferenceJob(
                    graph=graph,
                    kind=source.get("kind", "best_known_cut"),
                    config={},
                    seed=None,
                    source=source,
                )
            ]
        if policy["type"] == "none":
            return []
        return []

    def run_job(self, job):
        """Execute one reference job and return an independent record."""
        if not isinstance(job, ReferenceJob):
            raise TypeError("job must be a ReferenceJob.")
        started = perf_counter()
        base = {
            "schema_version": 1,
            "record_type": "reference",
            "reference_id": job.reference_id,
            "kind": job.kind,
            "graph_content_id": job.graph_content_id,
            "graph_instance_id": job.graph_instance_id,
            "node_order": list(job.graph.nodes),
            "config": to_json_data(job.config),
            "seed": job.seed,
            "status": "unavailable",
            "reason": None,
            "cut_value": None,
            "solution": None,
            "proof": None,
            "upper_bound": None,
            "source": to_json_data(job.source),
            "transformed_graph": None,
            "solve_seconds": None,
            "wall_seconds": None,
            "solver_result": None,
        }
        if job.kind == "best_known_cut":
            source = job.source or {}
            base.update(
                status="completed",
                cut_value=float(source["value"]),
                proof=source.get("proof", "external"),
                source=to_json_data(source),
                upper_bound=source.get("upper_bound"),
            )
            base["wall_seconds"] = perf_counter() - started
            return base
        from .factory import SolverFactory

        try:
            if job.kind in ("exact_max_cut", "exact_min_cut"):
                graph = job.graph
                if job.kind == "exact_min_cut":
                    graph = WeightedGraph(
                        [(u, v, -float(weight)) for u, v, weight in graph.edges],
                        nodes=graph.nodes,
                    )
                solver = SolverFactory.from_config(deepcopy(job.config))
                solution = solver.solve(graph, report=0, logs=0)
                record = solver.last_run
                if record is None or record.get("status") not in ("completed", "stopped"):
                    raise RuntimeError("Exact reference did not complete.")
                if record.get("termination_reason") != "exhaustive_search":
                    raise RuntimeError("Exact reference did not prove exhaustive search.")
                cut = graph.cut_value(solution)
                if job.kind == "exact_min_cut":
                    cut = -cut
                    transformed_id = graph_content_id(graph)
                else:
                    transformed_id = None
                base.update(
                    status="completed",
                    cut_value=float(cut),
                    solution={int(k): int(v) for k, v in solution.items()},
                    proof="exhaustive",
                    upper_bound=float(cut) if job.kind == "exact_max_cut" else None,
                    transformed_graph=transformed_id,
                    solve_seconds=record.get("timing", {}).get("compute_seconds"),
                    solver_result={
                        "algorithm_type": record.get("algorithm_type"),
                        "run_id": record.get("run_id"),
                        "status": record.get("status"),
                        "termination_reason": record.get("termination_reason"),
                        "worst_cut_value": record.get("worst_cut_value"),
                    },
                )
            elif job.kind == "gw_rounded_cut":
                solver = SolverFactory.from_config(deepcopy(job.config))
                solution = solver.solve(job.graph, report=0, logs=0)
                record = solver.last_run
                if record is None or record.get("status") not in ("completed", "stopped"):
                    raise RuntimeError("GW reference did not complete.")
                base.update(
                    status="completed",
                    cut_value=float(job.graph.cut_value(solution)),
                    solution={int(k): int(v) for k, v in solution.items()},
                    proof="heuristic",
                    upper_bound=None,
                    solve_seconds=record.get("timing", {}).get("compute_seconds"),
                    solver_result={
                        "algorithm_type": record.get("algorithm_type"),
                        "run_id": record.get("run_id"),
                        "status": record.get("status"),
                        "termination_reason": record.get("termination_reason"),
                        "sdp_objective_raw": record.get("sdp_objective_raw"),
                        "sdp_objective_repaired": record.get("sdp_objective_repaired"),
                        "sdp_upper_bound": record.get("sdp_upper_bound"),
                        "positive_weight_upper_bound": record.get("positive_weight_upper_bound"),
                    },
                )
            else:
                raise ValueError(f"Unsupported reference kind: {job.kind}")
        except Exception as error:
            base.update(
                status="failed",
                reason=f"{type(error).__name__}: {error}",
            )
        base["wall_seconds"] = perf_counter() - started
        return base

    def run_all(self, jobs):
        """Execute jobs in order and retain independent result records."""
        self.last_references = [self.run_job(job) for job in jobs]
        return deepcopy(self.last_references)
