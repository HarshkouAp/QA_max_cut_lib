"""Compatibility adapters for the experimental Laboratory variants.

R19 requires the materially different experimental branches of the historical
archive to remain reachable from the current API.  The archive used several
near-duplicate controllers with private NumPy arrays and graphs.  Rather than
copy those arrays and loops again, this module maps each branch to the shared
CPU controllers and records the recipe that distinguishes it from its siblings.

The adapters are deliberately small: they create a validated configuration,
delegate to ``LSRQAOACPU`` or ``BatchLSRQAOACPU``, and expose the same public
``solve()`` lifecycle.  Exact source names are kept in ``SOURCE_ALIASES`` for
audit and migration reports.
"""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from ..core.graph import WeightedGraph
from ..solvers.batch_ls import BatchLSRQAOACPU
from ..solvers.ls_rqaoa_cpu import LSRQAOACPU
from ..solvers.qaoa_cpu import QAOACPU

__all__ = [
    "SOURCE_ALIASES",
    "LegacyAdapterError",
    "CorrectedMeanMap",
    "CorrelationExtractor",
    "EdgeExtender",
    "GraphDecomposerLight",
    "GraphScanner",
    "LSRQAOAExp",
    "LSRQAOAFast1",
    "LSRQAOAFast2",
    "LSRQAOAFast3",
    "LSRQAOAFast4",
    "MK6V1",
    "MK6V2",
    "MK7V1",
    "MK7V2",
    "QAOAAdam",
    "QAOAF2",
    "QAOAFM",
    "partition_smoke_test",
]

SOURCE_ALIASES = {
    "Diplom/MK_6_1.py": "MK6V1",
    "Diplom/MK_6_2.py": "MK6V2",
    "Diplom/MK_7_1.py": "MK7V1",
    "Diplom/MK_7_2.py": "MK7V2",
    "Laboratory/correct_mean.py": "CorrectedMeanMap",
    "Laboratory/CorrelationExtractor.py": "CorrelationExtractor",
    "Laboratory/EdgeExtender.py": "EdgeExtender",
    "Laboratory/GraphDecomposer_light.py": "GraphDecomposerLight",
    "Laboratory/GraphScanning.py": "GraphScanner",
    "Laboratory/LS_RQAOA_exp.py": "LSRQAOAExp",
    "Laboratory/LS_RQAOA_fast_1.py": "LSRQAOAFast1",
    "Laboratory/LS_RQAOA_fast_2.py": "LSRQAOAFast2",
    "Laboratory/LS_RQAOA_fast_3.py": "LSRQAOAFast3",
    "Laboratory/LS_RQAOA_fast_4.py": "LSRQAOAFast4",
    "Laboratory/QAOA_ADAM.py": "QAOAAdam",
    "Laboratory/QAOA_F2.py": "QAOAF2",
    "Laboratory/QAOA_F_M.py": "QAOAFM",
    "Laboratory/test_partition.py": "partition_smoke_test",
}


class LegacyAdapterError(ValueError):
    """Raised when a legacy adapter receives an unsupported input."""


def _graph(value):
    if not isinstance(value, WeightedGraph):
        raise TypeError("graph must be a WeightedGraph.")
    return value


def _canonical(u, v):
    return (min(int(u), int(v)), max(int(u), int(v)))


def _analytic_source():
    return {
        "type": "qaoa_analytic_p1_cpu",
        "parameters": {"layers": 1, "dtype": "float64", "pair_batch_size": 128},
        "objective": {"type": "mean_cut", "parameters": {}},
        "objective_reference": "working_graph",
        "estimation": {"mode": "exact"},
        "optimizer": {
            "type": "gamma_grid_exact_beta",
            "parameters": {
                "gamma_bounds": [-math.pi, math.pi],
                "points": 5,
                "gamma_batch_size": 2,
            },
        },
        "correlations": {"mode": "exact"},
    }


def _ls_config(builder, *, seed=19, finish_at=2):
    return {
        "type": "ls_rqaoa_cpu",
        "seed": int(seed),
        "num_treads": 1,
        "parameters": {"finish_at": int(finish_at), "max_reductions": 2},
        "subgraph_builder": {
            "type": builder,
            "parameters": {
                "max_size": 3,
                "rounds": 1,
                "full_graph_when_fits": True,
                "boundary": "induced_no_fields",
            },
        },
        "correlation_source": _analytic_source(),
        "aggregation": {"type": "mean", "parameters": {}},
        "selection": {
            "type": "max_abs_correlation",
            "parameters": {"pairs": "edges", "zero_sign": 1},
        },
        "coverage": {"require_full": False, "empty_policy": "anchor_pair"},
        "update": {"type": "refresh_all"},
        "execution": {"backend": "process", "start_method": "spawn", "native_threads": 1},
        "budget_policy": {"type": "step_round_robin"},
        "finisher": {"type": "exact_cpu", "parameters": {"max_vertices": 8}},
        "limits": {"shots_total": 0, "max_local_tasks_total": 16},
        "collect": {
            "reduction_history": True,
            "parameter_trajectory": False,
            "local_task_history": False,
            "local_optimizer_history": False,
            "correlation_tables": False,
            "final_reduction_state": False,
        },
        "output": {"report": 0, "logs": 0, "color": "never"},
    }


def _batch_config(*, policy, max_size, seed=19):
    return {
        "type": "laboratory.batch_ls_rqaoa_cpu",
        "seed": int(seed),
        "num_treads": 1,
        "finish_at": 2,
        "max_reductions": 2,
        "batch": {
            "policy": policy,
            "proposal_order": "task_id",
            "pair_scope": "frozen_origin",
            "proposal_repetition": "last",
        },
        "initial_builder": {
            "type": "random_partition",
            "parameters": {
                "max_size": int(max_size),
                "rounds": 1,
                "full_graph_when_fits": True,
            },
        },
        "source": _analytic_source(),
        "finisher": {"type": "exact_cpu", "max_nodes": 4},
        "limits": {"max_parent_epochs": 2, "max_source_calls": 8, "max_shots": 4096},
        "collect": {"batch_history": False, "proposal_history": False, "reduction_history": False},
        "output": {"report": 0, "logs": 0, "color": "never"},
    }


class _Adapter:
    """Common lifecycle for a configured legacy recipe."""

    variant = "legacy"
    source = None
    solver_class = LSRQAOACPU

    def __init__(self, config=None):
        if config is not None:
            self.config = deepcopy(config)
        else:
            self.config = self.default_config()
        self._solver = None

    def default_config(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def solve(self, graph, **kwargs):
        self._solver = self.solver_class(self.config)
        return self._solver.solve(_graph(graph), **kwargs)

    @property
    def result(self):
        if self._solver is None:
            raise LegacyAdapterError("solve() has not been called.")
        return self._solver.result


class LSRQAOAExp(_Adapter):
    variant = "ls_rqaoa_exp"
    source = "Laboratory/LS_RQAOA_exp.py"
    solver_class = LSRQAOACPU

    def default_config(self):
        config = _ls_config("random_partition")
        config["parameters"]["max_reductions"] = 2
        return config


class LSRQAOAFast1(LSRQAOAExp):
    variant = "ls_rqaoa_fast_1"
    source = "Laboratory/LS_RQAOA_fast_1.py"

    def default_config(self):
        config = _ls_config("greedy_abs_weight_partition")
        config["subgraph_builder"]["parameters"]["rounds"] = 1
        return config


class LSRQAOAFast2(LSRQAOAExp):
    variant = "ls_rqaoa_fast_2"
    source = "Laboratory/LS_RQAOA_fast_2.py"

    def default_config(self):
        config = _ls_config("greedy_abs_weight_partition")
        config["subgraph_builder"]["parameters"]["max_size"] = 4
        config["subgraph_builder"]["parameters"]["rounds"] = 2
        return config


class LSRQAOAFast3(LSRQAOAExp):
    variant = "ls_rqaoa_fast_3"
    source = "Laboratory/LS_RQAOA_fast_3.py"

    def default_config(self):
        config = _ls_config("random_partition")
        config["subgraph_builder"]["parameters"]["max_size"] = 4
        config["subgraph_builder"]["parameters"]["rounds"] = 2
        return config


class LSRQAOAFast4(LSRQAOAExp):
    variant = "ls_rqaoa_fast_4"
    source = "Laboratory/LS_RQAOA_fast_4.py"

    def default_config(self):
        config = _ls_config("greedy_abs_weight_partition")
        config["subgraph_builder"]["parameters"]["max_size"] = 5
        config["subgraph_builder"]["parameters"]["rounds"] = 2
        return config


class _BatchVariant(_Adapter):
    solver_class = BatchLSRQAOACPU

    def __init__(self, config=None, *, policy="parity_forest", max_size=3):
        self._policy = policy
        self._max_size = max_size
        super().__init__(config)

    def default_config(self):
        return _batch_config(policy=self._policy, max_size=self._max_size)


class MK6V1(_BatchVariant):
    variant = "mk6v1"
    source = "Diplom/MK_6_1.py"


class MK6V2(_BatchVariant):
    variant = "mk6v2"
    source = "Diplom/MK_6_2.py"

    def default_config(self):
        return _batch_config(policy="active_endpoints_only", max_size=3)


class MK7V1(_BatchVariant):
    variant = "mk7v1"
    source = "Diplom/MK_7_1.py"

    def default_config(self):
        return _batch_config(policy="parity_forest", max_size=4)


class MK7V2(_BatchVariant):
    variant = "mk7v2"
    source = "Diplom/MK_7_2.py"

    def default_config(self):
        return _batch_config(policy="active_endpoints_only", max_size=4)


class CorrectedMeanMap:
    """Compute a mean map from states and probabilities without mutating input."""

    source = "Laboratory/correct_mean.py"

    def __init__(self, node_order=None):
        self.node_order = None if node_order is None else [int(n) for n in node_order]

    def __call__(self, states, probabilities=None, node_order=None):
        states = np.asarray(states, dtype=float)
        if states.ndim != 2:
            raise LegacyAdapterError("states must have shape (outcomes, nodes).")
        order = self.node_order if node_order is None else [int(n) for n in node_order]
        if order is None:
            order = list(range(states.shape[1]))
        if len(order) != states.shape[1]:
            raise LegacyAdapterError("node_order length does not match states.")
        if probabilities is None:
            probabilities = np.full(states.shape[0], 1.0 / states.shape[0])
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != (states.shape[0],):
            raise LegacyAdapterError("probabilities have the wrong length.")
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise LegacyAdapterError("probabilities must be finite and non-negative.")
        total = float(probabilities.sum())
        if total <= 0:
            raise LegacyAdapterError("probabilities must have positive mass.")
        means = (probabilities[:, None] * states).sum(axis=0) / total
        return {int(node): float(value) for node, value in zip(order, means)}


class CorrelationExtractor:
    """Extract pairwise ZZ correlations from a statevector or spin samples."""

    source = "Laboratory/CorrelationExtractor.py"

    def __init__(self, node_order=None):
        self.node_order = None if node_order is None else [int(n) for n in node_order]

    def from_statevector(self, amplitudes, node_order=None):
        amplitudes = np.asarray(amplitudes, dtype=complex).reshape(-1)
        n = int(round(math.log2(amplitudes.size)))
        if 2 ** n != amplitudes.size:
            raise LegacyAdapterError("statevector length must be a power of two.")
        order = self.node_order if node_order is None else [int(n) for n in node_order]
        if order is None:
            order = list(range(n))
        if len(order) != n:
            raise LegacyAdapterError("node_order length does not match statevector.")
        probs = np.abs(amplitudes) ** 2
        indices = np.arange(2 ** n)
        bits = ((indices[:, None] >> np.arange(n)) & 1) * 2 - 1
        matrix = (probs[:, None, None] * bits[:, :, None] * bits[:, None, :]).sum(axis=0)
        result = {}
        for i in range(n):
            for j in range(i + 1, n):
                result[_canonical(order[i], order[j])] = float(matrix[i, j])
        return result

    def from_samples(self, samples, node_order=None):
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2:
            raise LegacyAdapterError("samples must have shape (shots, nodes).")
        order = self.node_order if node_order is None else [int(n) for n in node_order]
        if order is None:
            order = list(range(samples.shape[1]))
        if len(order) != samples.shape[1]:
            raise LegacyAdapterError("node_order length does not match samples.")
        result = {}
        for i in range(samples.shape[1]):
            for j in range(i + 1, samples.shape[1]):
                result[_canonical(order[i], order[j])] = float(
                    np.mean(samples[:, i] * samples[:, j])
                )
        return result

    extract = from_samples


class EdgeExtender:
    """Build bounded induced subgraphs around an edge without legacy arrays."""

    source = "Laboratory/EdgeExtender.py"

    def __init__(self, max_size=4):
        self.max_size = int(max_size)
        if self.max_size < 2:
            raise LegacyAdapterError("max_size must be at least two.")

    def around(self, graph, edge):
        graph = _graph(graph)
        u, v = _canonical(*edge)
        if u not in graph.nodes or v not in graph.nodes:
            raise LegacyAdapterError("edge endpoints must belong to the graph.")
        chosen = [u, v]
        candidates = sorted(
            (node for node in graph.nodes if node not in chosen),
            key=lambda node: (-abs(graph.weight(node, u)) - abs(graph.weight(node, v)), int(node)),
        )
        for node in candidates:
            if len(chosen) >= self.max_size:
                break
            if graph.weight(node, u) != 0 or graph.weight(node, v) != 0:
                chosen.append(node)
        while len(chosen) < self.max_size:
            remaining = sorted(set(graph.nodes) - set(chosen))
            if not remaining:
                break
            chosen.append(remaining[0])
        return graph.subgraph(chosen)

    def all_edges(self, graph):
        graph = _graph(graph)
        return [self.around(graph, (u, v)) for u, v, _ in graph.edges]


class GraphDecomposerLight:
    """Generate disjoint induced blocks with a typed current-plan builder."""

    source = "Laboratory/GraphDecomposer_light.py"

    def __init__(self, max_size=4, rounds=1, seed=17):
        self.max_size = int(max_size)
        self.rounds = int(rounds)
        self.seed = seed

    def decompose(self, graph):
        graph = _graph(graph)
        from ..generators.subgraphs import build_subgraph_builder

        builder = build_subgraph_builder(
            {
                "type": "random_partition",
                "parameters": {
                    "max_size": self.max_size,
                    "rounds": self.rounds,
                    "full_graph_when_fits": True,
                },
            },
            seed=self.seed,
        )
        tasks = builder.build(graph, version=0)
        return [graph.subgraph(task["nodes"]) for task in tasks]

    generate_multiple_subgraphs = decompose


class GraphScanner:
    """Scan exact cuts over a cover and return the best feasible scan."""

    source = "Laboratory/GraphScanning.py"

    def __init__(self, max_size=4, seed=17):
        self.decomposer = GraphDecomposerLight(max_size=max_size, seed=seed)

    def scan(self, graph):
        from ..solvers.exact_cpu import ExactCPU

        graph = _graph(graph)
        best = None
        for block in self.decomposer.decompose(graph):
            solver = ExactCPU()
            solution = solver.solve(block)
            cut = block.cut_value(solution)
            if best is None or cut > best["cut"]:
                best = {"cut": cut, "solution": solution, "block": block}
        return best


class _QAOAAdapter(_Adapter):
    solver_class = QAOACPU

    def default_config(self):
        return {
            "seed": 19,
            "parameters": {"layers": 1, "dtype": "complex128"},
            "objective": {"type": "mean_cut", "parameters": {}},
            "estimation": {"mode": "exact"},
            "optimizer": {"type": "adam", "parameters": {"max_steps": 2}},
            "initialization": {"type": "fixed_angles", "parameters": {"values": [0.3, 0.2]}},
            "readout": {"mode": "most_probable"},
            "limits": {"shots_total": 0},
            "output": {"report": 0, "logs": 0, "color": "never"},
        }


class QAOAAdam(_QAOAAdapter):
    variant = "qaoa_adam"
    source = "Laboratory/QAOA_ADAM.py"


class QAOAF2(_QAOAAdapter):
    variant = "qaoa_f2"
    source = "Laboratory/QAOA_F2.py"


class QAOAFM(_QAOAAdapter):
    variant = "qaoa_fm"
    source = "Laboratory/QAOA_F_M.py"


def partition_smoke_test(graph, max_size=3, seed=17):
    """Replace the archived ``test_partition.py`` smoke check with a typed audit."""

    graph = _graph(graph)
    decomposer = GraphDecomposerLight(max_size=max_size, seed=seed)
    blocks = decomposer.decompose(graph)
    covered = set()
    for block in blocks:
        covered.update(block.nodes)
    if not covered <= set(graph.nodes):
        raise LegacyAdapterError("partition covers a vertex outside the graph.")
    return {"blocks": len(blocks), "covered": sorted(covered), "complete": covered == set(graph.nodes)}
