"""Capability adapter between typed plans and local QAOA tasks.

The existing LS-RQAOA path is induced-only.  This module builds an exact
edge-subset payload only when a typed plan explicitly declares that local
Hamiltonian; it does not convert an edge partition into an induced task.
"""

from __future__ import annotations

from copy import deepcopy

from ..core.graph import WeightedGraph
from ..generators.plans import SubproblemSpec, SubproblemPlan
from .ls_worker import solve_local_task

__all__ = [
    "LocalProblemCapabilityError",
    "local_graph_for_spec",
    "validate_capability",
    "build_edge_subset_payload",
    "run_edge_subset_task",
]


class LocalProblemCapabilityError(ValueError):
    """Raised when a solver cannot represent a plan's edge domain."""


def local_graph_for_spec(graph, spec):
    if not isinstance(spec, SubproblemSpec):
        raise TypeError("spec must be a SubproblemSpec.")
    if spec.edge_domain == "induced":
        return graph.subgraph(spec.nodes)
    return WeightedGraph(
        nodes=spec.nodes,
        edges=[(u, v, weight) for u, v, weight in spec.edges],
    )


def validate_capability(plan, local_problem="induced_no_fields"):
    if isinstance(plan, dict):
        plan = SubproblemPlan.from_dict(plan)
    if local_problem not in ("induced_no_fields", "edge_subset_no_fields"):
        raise LocalProblemCapabilityError("unknown local problem boundary.")
    if plan.plan_kind == "edge_partition" and local_problem != "edge_subset_no_fields":
        raise LocalProblemCapabilityError(
            "edge_partition requires local_problem='edge_subset_no_fields'."
        )
    if plan.plan_kind in ("vertex_partition", "induced_cover") and local_problem == "edge_subset_no_fields":
        raise LocalProblemCapabilityError(
            "induced plans need the induced_no_fields adapter."
        )
    return True


def build_edge_subset_payload(
    graph,
    spec,
    *,
    source_config,
    seed,
    quota,
    training_steps,
    parent_version=0,
    collect_optimizer_history=False,
    collect_objective_history=False,
):
    validate_capability(
        SubproblemPlan(
            plan_kind="edge_partition",
            parent_hash=graph.snapshot()["fingerprint"],
            builder_recipe={},
            parts=(spec,),
            completion_status="complete",
        ),
        "edge_subset_no_fields",
    )
    if spec.edge_domain != "explicit":
        raise LocalProblemCapabilityError("edge-subset payload needs explicit edge domain.")
    return {
        "task_id": spec.task_id,
        "nodes": [int(node) for node in spec.nodes],
        "edges": [[int(u), int(v), float(w)] for u, v, w in spec.edges],
        "parent_version": int(parent_version),
        "source_config": deepcopy(source_config),
        "seed": seed,
        "pair_domain": "edges",
        "shot_quota": int(quota),
        "training_steps": int(training_steps),
        "collect_optimizer_history": bool(collect_optimizer_history),
        "collect_objective_history": bool(collect_objective_history),
    }


def run_edge_subset_task(graph, spec, *, source_config, seed=0, quota=0, training_steps=0):
    payload = build_edge_subset_payload(
        graph,
        spec,
        source_config=source_config,
        seed=seed,
        quota=quota,
        training_steps=training_steps,
    )
    return solve_local_task(payload)
