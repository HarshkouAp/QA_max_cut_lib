"""Passive mutual-information observer assemblies for QAOA and RQAOA."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import math

import numpy as np

from ..core.graph import WeightedGraph
from ..tools.mutual_information import (
    MutualInformationError,
    classical_from_counts,
    classical_from_probabilities,
    classical_mutual_information,
    hop_distances,
    map_summary,
    pair_scope,
    quantum_mutual_information,
    reduced_density_matrix,
    validate_node_order,
    validate_statevector,
)
from .components import QAOAStatevectorCorrelationSource
from .qaoa_cpu import QAOACPU
from .qaoa_state import cut_diagonal, state_at_angles
from .rqaoa_cpu import RQAOACPU

__all__ = [
    "MutualInformationConfigurationError",
    "MutualInformationObserver",
    "QAOAWithMIObserverCPU",
    "RQAOAWithMIObserverCPU",
    "RecordingCorrelationSource",
]


class MutualInformationConfigurationError(ValueError):
    """Raised for an invalid passive-observer configuration."""


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise MutualInformationConfigurationError(f"{name} must be finite.")
    return float(value)


class MutualInformationObserver:
    """Validate and evaluate one passive pair-information plan."""

    def __init__(self, config):
        config = deepcopy(config or {})
        if set(config) - {
            "quantum",
            "classical_z",
            "snapshots",
            "pairs",
            "coordinates",
            "pair_batch_size",
            "analysis_limits",
            "distance_profile",
            "collect",
        }:
            raise MutualInformationConfigurationError("Unknown mutual_information field.")
        quantum = config.setdefault(
            "quantum", {"enabled": True, "definition": "standard", "log_base": 2, "legacy_half_nats_view": False}
        )
        if set(quantum) - {"enabled", "definition", "log_base", "legacy_half_nats_view"}:
            raise MutualInformationConfigurationError("Unknown quantum MI field.")
        quantum.setdefault("enabled", True)
        quantum.setdefault("definition", "standard")
        quantum.setdefault("log_base", 2)
        quantum.setdefault("legacy_half_nats_view", False)
        if type(quantum["enabled"]) is not bool:
            raise MutualInformationConfigurationError("quantum.enabled must be boolean.")
        if quantum["definition"] != "standard":
            raise MutualInformationConfigurationError("Only standard QMI is implemented.")
        if quantum["log_base"] not in (2, "e", math.e):
            raise MutualInformationConfigurationError("quantum.log_base must be 2 or e.")
        if type(quantum["legacy_half_nats_view"]) is not bool:
            raise MutualInformationConfigurationError("legacy_half_nats_view must be boolean.")
        classical = config.setdefault(
            "classical_z", {"enabled": True, "log_base": 2, "estimator": "probabilities_exact"}
        )
        if set(classical) - {"enabled", "log_base", "estimator"}:
            raise MutualInformationConfigurationError("Unknown classical_z field.")
        classical.setdefault("enabled", True)
        classical.setdefault("log_base", 2)
        classical.setdefault("estimator", "probabilities_exact")
        if type(classical["enabled"]) is not bool:
            raise MutualInformationConfigurationError("classical_z.enabled must be boolean.")
        if classical["log_base"] not in (2, "e", math.e):
            raise MutualInformationConfigurationError("classical_z.log_base must be 2 or e.")
        if classical["estimator"] not in ("probabilities_exact", "reuse_readout_batch", "reuse_correlation_batch"):
            raise MutualInformationConfigurationError("Unknown classical Z estimator.")
        snapshots = config.setdefault(
            "snapshots", {"type": "selected_final", "include_initial": False}
        )
        if set(snapshots) - {"type", "include_initial"}:
            raise MutualInformationConfigurationError("Unknown snapshots field.")
        snapshots.setdefault("type", "selected_final")
        snapshots.setdefault("include_initial", False)
        if snapshots["type"] not in ("selected_final", "final_angle_prefixes"):
            raise MutualInformationConfigurationError("Unknown snapshot type.")
        if type(snapshots["include_initial"]) is not bool:
            raise MutualInformationConfigurationError("include_initial must be boolean.")
        pairs = config.setdefault("pairs", {"type": "all_pairs"})
        if set(pairs) - {"type", "values"}:
            raise MutualInformationConfigurationError("Unknown pairs field.")
        if pairs.setdefault("type", "all_pairs") not in ("all_pairs", "edges", "explicit"):
            raise MutualInformationConfigurationError("Unknown pair scope.")
        if pairs["type"] == "explicit":
            if not isinstance(pairs.get("values"), list) or not pairs["values"]:
                raise MutualInformationConfigurationError("explicit pairs require values.")
        config.setdefault("coordinates", "active")
        if config["coordinates"] != "active":
            raise MutualInformationConfigurationError("Only active coordinates are implemented.")
        config.setdefault("pair_batch_size", 8)
        if isinstance(config["pair_batch_size"], bool) or not isinstance(config["pair_batch_size"], int) or config["pair_batch_size"] < 1:
            raise MutualInformationConfigurationError("pair_batch_size must be positive.")
        limits = config.setdefault(
            "analysis_limits",
            {"max_qubits": 18, "max_pairs_per_snapshot": 153, "max_snapshots": 100, "max_pair_evaluations_total": 10000},
        )
        if set(limits) - {"max_qubits", "max_pairs_per_snapshot", "max_snapshots", "max_pair_evaluations_total"}:
            raise MutualInformationConfigurationError("Unknown analysis limit.")
        for key, value in limits.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise MutualInformationConfigurationError(f"analysis_limits.{key} must be positive.")
        distance = config.setdefault("distance_profile", {"type": "hop", "graph": "snapshot_current"})
        if set(distance) - {"type", "graph"} or distance.get("type") != "hop" or distance.get("graph") != "snapshot_current":
            raise MutualInformationConfigurationError("Only hop/snapshot_current distance profile is implemented.")
        collect = config.setdefault("collect", {"one_qubit_rdms": False, "pair_rdms": False})
        if set(collect) - {"one_qubit_rdms", "pair_rdms"}:
            raise MutualInformationConfigurationError("Unknown MI collect field.")
        for key in ("one_qubit_rdms", "pair_rdms"):
            collect.setdefault(key, False)
            if type(collect[key]) is not bool:
                raise MutualInformationConfigurationError(f"collect.{key} must be boolean.")
        self.config = config

    def _preflight_pairs(self, labels, graph):
        pairs = pair_scope(labels, graph, self.config["pairs"]["type"], self.config["pairs"].get("values"))
        limits = self.config["analysis_limits"]
        if len(labels) > limits["max_qubits"]:
            raise MutualInformationConfigurationError("Snapshot exceeds max_qubits.")
        if len(pairs) > limits["max_pairs_per_snapshot"]:
            raise MutualInformationConfigurationError("Requested pairs exceed max_pairs_per_snapshot.")
        return pairs

    def _snapshot_states(self, graph, node_order, theta, layers, dtype):
        cuts = cut_diagonal(graph)
        snapshots = self.config["snapshots"]
        if snapshots["type"] == "selected_final":
            state, probabilities = state_at_angles(cuts, theta, layers, dtype=dtype)
            return [{"layer": int(layers), "kind": "selected_final", "state": state, "probabilities": probabilities}]
        result = []
        if snapshots["include_initial"]:
            size = 1 << len(node_order)
            state = np.ones(size, dtype=np.complex128 if dtype == "complex128" else np.complex64) / math.sqrt(size)
            result.append({"layer": 0, "kind": "initial", "state": state, "probabilities": np.abs(state) ** 2})
        theta = np.asarray(theta, dtype=float)
        for layer in range(1, int(layers) + 1):
            prefix = np.r_[theta[:layer], theta[int(layers): int(layers) + layer]]
            state, probabilities = state_at_angles(cuts, prefix, layer, dtype=dtype)
            result.append({"layer": int(layer), "kind": "final_angle_prefix", "state": state, "probabilities": probabilities})
        return result

    def evaluate(self, graph, node_order, theta, layers, dtype, *, readout_batch=None):
        """Return passive MI frames for one frozen state/graph."""
        labels = validate_node_order(node_order, len(node_order))
        if not isinstance(graph, WeightedGraph):
            raise TypeError("graph must be a WeightedGraph.")
        if self.config["snapshots"]["type"] == "final_angle_prefixes":
            snapshot_count = (1 if self.config["snapshots"]["include_initial"] else 0) + int(layers)
        else:
            snapshot_count = 1
        if snapshot_count > self.config["analysis_limits"]["max_snapshots"]:
            raise MutualInformationConfigurationError("Requested snapshots exceed max_snapshots.")
        estimated_pairs = self._preflight_pairs(labels, graph)
        if snapshot_count * len(estimated_pairs) > self.config["analysis_limits"]["max_pair_evaluations_total"]:
            raise MutualInformationConfigurationError("Pair evaluations exceed max_pair_evaluations_total.")
        states = self._snapshot_states(graph, labels, theta, layers, dtype)
        qcfg = self.config["quantum"]
        zcfg = self.config["classical_z"]
        frames = []
        for snapshot in states:
            state = validate_statevector(snapshot["state"], labels)
            probabilities = np.abs(state) ** 2
            pairs = pair_scope(labels, graph, self.config["pairs"]["type"], self.config["pairs"].get("values"))
            qvalues = {}
            zvalues = {}
            one = {}
            sampling_estimator = zcfg["estimator"] in (
                "reuse_readout_batch",
                "reuse_correlation_batch",
            )
            if sampling_estimator and readout_batch is None:
                raise MutualInformationConfigurationError(
                    "sample-reuse MI requires an actual sampled batch."
                )
            pair_rdms = {}
            for pair in pairs:
                i, j = pair
                qmi = quantum_mutual_information(state, labels, pair, base=qcfg["log_base"])
                if sampling_estimator:
                    batch = np.asarray(readout_batch, dtype=np.int64)
                    ii = labels.index(i)
                    jj = labels.index(j)
                    bi = (batch >> ii) & 1
                    bj = (batch >> jj) & 1
                    table = np.zeros((2, 2), dtype=float)
                    np.add.at(table, (bi, bj), 1.0)
                    zmi = classical_from_counts(table, base=zcfg["log_base"])
                else:
                    zmi = classical_from_probabilities(probabilities, labels, pair, base=zcfg["log_base"])
                qvalues[pair] = float(qmi)
                zvalues[pair] = float(zmi)
                if self.config["collect"]["one_qubit_rdms"]:
                    one[i] = reduced_density_matrix(state, [labels.index(i)], labels).tolist()
                    one[j] = reduced_density_matrix(state, [labels.index(j)], labels).tolist()
                if self.config["collect"]["pair_rdms"]:
                    pair_rdms[pair] = reduced_density_matrix(
                        state, [labels.index(i), labels.index(j)], labels
                    ).tolist()
            qsummary = map_summary(
                np.array(list(qvalues.values()), dtype=float),
                np.ones(len(qvalues), dtype=bool),
            )
            zsummary = map_summary(
                np.array(list(zvalues.values()), dtype=float),
                np.ones(len(zvalues), dtype=bool),
            )
            distances, disconnected = hop_distances(graph, labels)
            profile = {}
            for pair, value in qvalues.items():
                distance = distances.get(pair)
                key = "disconnected" if distance is None else str(int(distance))
                profile.setdefault(key, []).append(value)
            distance_profile = {key: {"count": len(values), "mean": float(np.mean(values))} for key, values in profile.items()}
            matrix_labels = [int(node) for node in labels]
            matrix_q = [[0.0 for _ in matrix_labels] for _ in matrix_labels]
            matrix_z = [[0.0 for _ in matrix_labels] for _ in matrix_labels]
            matrix_mask = [[False for _ in matrix_labels] for _ in matrix_labels]
            for (u, v), value in qvalues.items():
                a = matrix_labels.index(int(u))
                b = matrix_labels.index(int(v))
                matrix_q[a][b] = matrix_q[b][a] = float(value)
                matrix_z[a][b] = matrix_z[b][a] = float(zvalues[(u, v)])
                matrix_mask[a][b] = matrix_mask[b][a] = True
            frame = {
                "layer": int(snapshot["layer"]),
                "kind": snapshot["kind"],
                "node_order": [int(node) for node in labels],
                "active_labels": [int(node) for node in labels],
                "pairs": [[int(u), int(v)] for u, v in pairs],
                "quantum_mi": [[int(u), int(v), float(qvalues[(u, v)])] for u, v in pairs],
                "classical_z_mi": [[int(u), int(v), float(zvalues[(u, v)])] for u, v in pairs],
                "quantum_summary": qsummary,
                "classical_z_summary": zsummary,
                "valid_mask": matrix_mask,
                "matrix": {
                    "labels": matrix_labels,
                    "quantum_mi": matrix_q,
                    "classical_z_mi": matrix_z,
                    "valid_mask": matrix_mask,
                },
                "distance_profile": distance_profile,
                "disconnected_pairs": [list(pair) for pair in disconnected],
                "source": "statevector_exact",
                "state_norm": float(np.linalg.norm(state)),
            }
            if qcfg["legacy_half_nats_view"]:
                frame["legacy_half_nats"] = [
                    [int(u), int(v), float(0.5 * quantum_mutual_information(state, labels, (u, v), base="e"))]
                    for u, v in pairs
                ]
            if self.config["collect"]["one_qubit_rdms"]:
                frame["one_qubit_rdms"] = one
            if self.config["collect"]["pair_rdms"]:
                frame["pair_rdms"] = pair_rdms
            frames.append(frame)
        return {
            "definition": "standard_quantum_mutual_information",
            "classical_definition": "classical_z_mutual_information",
            "log_base": qcfg["log_base"],
            "classical_log_base": zcfg["log_base"],
            "units": "bits" if qcfg["log_base"] == 2 else "nats",
            "legacy_half_nats_view": bool(qcfg["legacy_half_nats_view"]),
            "snapshot_policy": self.config["snapshots"]["type"],
            "frames": frames,
            "frame_count": len(frames),
            "pair_scope": self.config["pairs"]["type"],
            "valid_pair_count": len(frames[0]["pairs"]) if frames else 0,
            "origin": "statevector_exact_simulator_only",
        }


class QAOAWithMIObserverCPU(QAOACPU):
    """QAOACPU with a passive post-training MI observer."""

    algorithm_type = "laboratory.qaoa_mi_cpu"
    extra_config_fields = QAOACPU.extra_config_fields + ("diagnostics",)
    optional_fields = QAOACPU.optional_fields + (
        "mutual_information",
        "mi_frames",
        "mi_summary",
    )

    def __init__(self, config=None):
        config = deepcopy(config or {})
        every = config.get("checkpoint", {}).get("every")
        if isinstance(every, dict):
            config["checkpoint"]["every"] = {
                key: value for key, value in every.items() if key == "optimizer_iterations"
            }
        super().__init__(config)

    def _validate(self, config, graph):
        super()._validate(config, graph)
        diagnostics = config.get("diagnostics")
        if not isinstance(diagnostics, dict) or "mutual_information" not in diagnostics:
            raise MutualInformationConfigurationError("diagnostics.mutual_information is required.")
        self._mi_observer = MutualInformationObserver(diagnostics["mutual_information"])
        if self._mi_observer.config["classical_z"]["estimator"] == "reuse_readout_batch":
            if config["readout"].get("mode") != "best_of_shots" or not config["readout"].get("shots"):
                raise MutualInformationConfigurationError(
                    "reuse_readout_batch needs best_of_shots readout with positive shots."
                )
            if self._mi_observer.config["snapshots"]["type"] != "selected_final":
                raise MutualInformationConfigurationError(
                    "reuse_readout_batch is valid for selected_final only."
                )
        self._mi_observer._preflight_pairs(tuple(graph.nodes), graph)

    def _run(self, graph, config, rng):
        fields = super()._run(graph, config, rng)
        if fields.get("status") != "completed":
            return fields
        theta = fields.get("optimal_angles")
        if theta is None:
            raise MutualInformationError("QAOA did not return selected final angles.")
        payload = self._mi_observer.evaluate(
            graph,
            tuple(graph.nodes),
            np.asarray(theta, dtype=float),
            int(config["parameters"]["layers"]),
            config["parameters"]["dtype"],
            readout_batch=getattr(self, "_last_readout_batch", None),
        )
        fields["mutual_information"] = payload
        fields["mi_frames"] = payload["frames"]
        fields["mi_summary"] = {
            "snapshot_count": payload["frame_count"],
            "quantum": [frame["quantum_summary"] for frame in payload["frames"]],
            "classical_z": [frame["classical_z_summary"] for frame in payload["frames"]],
        }
        return fields


class RecordingCorrelationSource:
    """Wrap a full-state source and freeze its state before each reduction."""

    def __init__(self, source):
        self.source = source
        self.frames = []

    def estimate(self, state, **kwargs):
        result = self.source.estimate(state, **kwargs)
        graph = state.current_graph()
        cuts = cut_diagonal(graph)
        angles = np.asarray(result.angles, dtype=float)
        statevector, probabilities = state_at_angles(
            cuts, angles, int(result.layers), dtype="complex128"
        )
        self.frames.append(
            {
                "graph": graph.copy(),
                "node_order": tuple(result.node_order),
                "task_version": int(result.task_version),
                "angles": angles.copy(),
                "layers": int(result.layers),
                "state": statevector,
                "probabilities": probabilities,
                "correlations": dict(result.values),
                "batch": getattr(self.source, "_last_samples", None),
                "cut_offset": getattr(state, "cut_offset", None),
                "task": state,
            }
        )
        return result


class RQAOAWithMIObserverCPU(RQAOACPU):
    """RQAOA with passive MI frames captured before each reduction."""

    algorithm_type = "laboratory.rqaoa_mi_cpu"
    extra_config_fields = RQAOACPU.extra_config_fields + ("diagnostics",)
    supports_checkpoint = False
    optional_fields = RQAOACPU.optional_fields + (
        "mutual_information",
        "mi_frames",
        "mi_summary",
    )

    def _validate(self, config, graph):
        super()._validate(config, graph)
        diagnostics = config.get("diagnostics")
        if not isinstance(diagnostics, dict) or "mutual_information" not in diagnostics:
            raise MutualInformationConfigurationError("diagnostics.mutual_information is required.")
        if config["correlation_source"].get("type") != "qaoa_statevector_cpu":
            raise MutualInformationConfigurationError(
                "RQAOA MI observer needs a full qaoa_statevector_cpu source."
            )
        self._mi_observer = MutualInformationObserver(diagnostics["mutual_information"])
        if self._mi_observer.config["snapshots"]["type"] != "selected_final":
            raise MutualInformationConfigurationError(
                "RQAOA frames support selected_final snapshots only."
            )
        if self._mi_observer.config["classical_z"]["estimator"] not in (
            "probabilities_exact",
            "reuse_correlation_batch",
        ):
            raise MutualInformationConfigurationError("Unsupported RQAOA classical Z estimator.")
        self._mi_observer._preflight_pairs(tuple(graph.nodes), graph)

    def _run(self, graph, config, rng):
        base_source = QAOAStatevectorCorrelationSource(
            config["correlation_source"], seed=config.get("seed")
        )
        recorder = RecordingCorrelationSource(base_source)
        self._injected_source = recorder
        try:
            fields = super()._run(graph, config, rng)
        finally:
            self._injected_source = None
        frames = []
        for item in recorder.frames:
            observer = self._mi_observer
            payload = observer.evaluate(
                item["graph"],
                item["node_order"],
                item["angles"],
                item["layers"],
                "complex128",
                readout_batch=item["batch"],
            )
            frame = payload["frames"][0]
            frame["reduction_step"] = item["task_version"] + 1
            frame["task_version"] = item["task_version"]
            frame["coordinates"] = "active_before_reduction"
            frame["cut_offset"] = item["cut_offset"]
            frame["removed_labels"] = [
                int(constraint.removed) for constraint in item["task"].constraints
            ]
            frame["original_labels"] = [int(label) for label in item["task"].original_nodes]
            original_labels = [int(label) for label in item["task"].original_nodes]
            original_values = [[0.0 for _ in original_labels] for _ in original_labels]
            original_mask = [[False for _ in original_labels] for _ in original_labels]
            for pair_entry in frame["quantum_mi"]:
                u, v, value = pair_entry
                a = original_labels.index(int(u))
                b = original_labels.index(int(v))
                original_values[a][b] = original_values[b][a] = float(value)
                original_mask[a][b] = original_mask[b][a] = True
            frame["original_matrix"] = {
                "labels": original_labels,
                "quantum_mi": original_values,
                "valid_mask": original_mask,
            }
            if observer.config["classical_z"]["estimator"] == "reuse_correlation_batch":
                # The source already produced a physical ZZ table for this frame.
                frame["classical_z_provenance"] = "source_correlation_batch"
                frame["source_correlations"] = [
                    [int(u), int(v), float(value)] for (u, v), value in item["correlations"].items()
                ]
            frames.append(frame)
        fields["mutual_information"] = {
            "definition": "standard_quantum_mutual_information",
            "classical_definition": "classical_z_mutual_information",
            "log_base": self._mi_observer.config["quantum"]["log_base"],
            "units": "bits" if self._mi_observer.config["quantum"]["log_base"] == 2 else "nats",
            "snapshot_policy": "selected_final_before_reduction",
            "frames": frames,
            "frame_count": len(frames),
            "origin": "statevector_exact_simulator_only",
        }
        fields["mi_frames"] = frames
        fields["mi_summary"] = {
            "snapshot_count": len(frames),
            "quantum": [frame["quantum_summary"] for frame in frames],
            "classical_z": [frame["classical_z_summary"] for frame in frames],
        }
        return fields


def main(argv=None):
    """Run the QAOA MI observer example through the common CLI."""
    from ..tools.demo import run_solver_demo

    return run_solver_demo(QAOAWithMIObserverCPU, argv)
