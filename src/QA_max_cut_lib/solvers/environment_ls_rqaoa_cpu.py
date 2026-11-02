"""LS-RQAOA with fixed-spin conditional local problems."""

from __future__ import annotations

from copy import deepcopy

from ..core.graph import WeightedGraph
from .conditional import ConditionalProblemError, build_conditional_problem, validate_reference
from .ls_rqaoa_cpu import LSRQAOACPU

__all__ = ["EnvironmentLSRQAOACPU"]


class EnvironmentLSRQAOACPU(LSRQAOACPU):
    """Use the shared LS controller with one frozen reference per step.

    Each local block is an exact fixed-spin conditional problem.  The global
    reducer remains zero-field; local fields are not copied into the parent
    graph.  After a reduction, the reference retains the survivor spin and
    removes the deleted label (``retain_survivor``).
    """

    algorithm_type = "laboratory.environment_ls_rqaoa_cpu"
    extra_config_fields = LSRQAOACPU.extra_config_fields + (
        "environment",
        "boundary",
        "fallback",
    )

    def _validate(self, config, graph):
        environment = config.setdefault("environment", {"type": "fixed_spins", "source": {"type": "all_plus"}})
        if set(environment) - {"type", "source", "update_policy", "epoch_policy"}:
            raise ValueError("Unknown environment field.")
        if environment.get("type") != "fixed_spins":
            raise ValueError("environment.type must be fixed_spins.")
        environment.setdefault("source", {"type": "all_plus"})
        environment.setdefault("update_policy", "retain_survivor")
        environment.setdefault("epoch_policy", "frozen_per_reduction")
        if environment["update_policy"] != "retain_survivor":
            raise ValueError("Only retain_survivor is implemented.")
        if environment["epoch_policy"] != "frozen_per_reduction":
            raise ValueError("Only frozen_per_reduction is implemented.")
        source = environment["source"]
        if source.get("type") not in ("all_plus", "provided_spins"):
            raise ValueError("environment source must be all_plus or provided_spins.")
        if source.get("type") == "provided_spins":
            params = source.setdefault("parameters", {})
            if set(params) != {"entries"} or not isinstance(params["entries"], list):
                raise ValueError("provided_spins requires entries.")
        boundary = config.setdefault("boundary", {"type": "fixed_spins_conditional"})
        if set(boundary) - {"type"} or boundary.get("type") != "fixed_spins_conditional":
            raise ValueError("Only fixed_spins_conditional is implemented.")
        fallback = config.setdefault("fallback", {"type": "reference_assignment"})
        if set(fallback) - {"type"} or fallback.get("type") != "reference_assignment":
            raise ValueError("Only reference_assignment fallback is implemented.")
        super()._validate(config, graph)
        config["correlation_source"]["type"] = "qaoa_ising_statevector_cpu"
        config["environment"] = environment
        config["boundary"] = boundary
        config["fallback"] = fallback

    def _prepare_reference(self, graph, config):
        source = config["environment"]["source"]
        if source["type"] == "all_plus":
            reference = {int(node): 1 for node in graph.nodes}
        else:
            reference = {}
            for item in source["parameters"]["entries"]:
                if not isinstance(item, dict) or set(item) - {"node", "spin"}:
                    raise ConditionalProblemError("Each reference entry needs node and spin.")
                node = item.get("node")
                if isinstance(node, bool) or not isinstance(node, int):
                    raise ConditionalProblemError("Reference nodes must be integers.")
                if node in reference:
                    raise ConditionalProblemError(f"Duplicate reference node: {node}")
                reference[node] = item.get("spin")
            reference = validate_reference(graph, reference)
        self._environment_reference = reference
        return reference

    def _run(self, graph, config, rng):
        self._prepare_reference(graph, config)
        fields = super()._run(graph, config, rng)
        fields["reference"] = {
            int(k): int(v) for k, v in self._environment_reference.items()
        }
        fields["final_nodes"] = [int(node) for node in self._environment_reference]
        return fields

    def _plan_tasks(self, state, config):
        tasks, allowed, observed, coverage = super()._plan_tasks(state, config)
        graph = state.current_graph()
        for task in tasks:
            problem = build_conditional_problem(
                graph,
                task["nodes"],
                self._environment_reference,
                cut_offset=float(state.cut_offset or 0.0),
            )
            task["conditional"] = problem.to_dict()
            task["local_problem"] = problem
        return tasks, allowed, observed, coverage

    def _make_payload(self, task, state, config, steps, quota):
        payload = super()._make_payload(task, state, config, steps, quota)
        payload["conditional"] = deepcopy(task["conditional"])
        payload["source_config"] = deepcopy(config["correlation_source"])
        return payload

    def _fallback_solution(self, state):
        reference = dict(self._environment_reference)
        # The reference is kept on the current active set after every reduction.
        missing = set(state.current_nodes) - set(reference)
        if missing:
            return {int(node): 1 for node in state.current_nodes}
        return {int(node): int(reference[node]) for node in state.current_nodes}

    def _on_reduction(self, state, event):
        reference = dict(self._environment_reference)
        removed = int(event["removed_node"])
        retained = int(event["retained_node"])
        if removed not in reference or retained not in reference:
            raise ConditionalProblemError("Reference is missing a reducer label.")
        del reference[removed]
        self._environment_reference = reference


def main(argv=None):
    from ..tools.demo import run_solver_demo

    return run_solver_demo(EnvironmentLSRQAOACPU, argv)


if __name__ == "__main__":
    raise SystemExit(main())
