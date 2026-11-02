"""Declared checkpoint boundaries for public implementations.

The table is intentionally explicit: a component that does not have a
versioned state codec is rejected instead of appearing resumable.
"""

from __future__ import annotations

__all__ = ["CAPABILITIES", "capability", "capabilities"]

CAPABILITIES = {
    "qaoa_cpu": {
        "supports_checkpoint": True,
        "boundaries": ["completed ADAM iteration", "training_done", "readout_done"],
        "state": "theta, m, v, best evaluation, RNG, budget, histories",
        "phases": ["training", "training_done", "readout_done"],
        "notes": "Full CPU QAOACPU codec, including central-difference probes and final readout token.",
    },
    "laboratory.warm_start_qaoa_cpu": {
        "supports_checkpoint": True,
        "boundaries": ["completed ADAM iteration", "training_done", "readout_done"],
        "state": "theta, m, v, best evaluation, RNG, budget, frozen source/q/mixer",
        "phases": ["training", "training_done", "readout_done"],
        "notes": "CPU full-state warm-start codec; source preparation is frozen in the checkpoint.",
    },
    "laboratory.conditional_qaoa_cpu": {
        "supports_checkpoint": False,
        "boundaries": [],
        "state": "not implemented",
        "phases": [],
        "notes": "Conditional local problem and external reference are not yet in a versioned codec; no resume is claimed.",
    },
    "laboratory.environment_ls_rqaoa_cpu": {
        "supports_checkpoint": False,
        "boundaries": [],
        "state": "not implemented",
        "phases": [],
        "notes": "Parent/environment epochs and conditional child tasks need a task codec; no resume is claimed.",
    },
    "laboratory.warm_start_rqaoa_cpu": {
        "supports_checkpoint": False,
        "boundaries": [],
        "state": "not implemented",
        "phases": [],
        "notes": "Ensemble child aggregation and warm-start source are not yet in a versioned codec.",
    },
    "laboratory.block_mean_field_qaoa_cpu": {
        "supports_checkpoint": True,
        "boundaries": ["complete sweep commit"],
        "state": "graph/partition, current m/q/theta bank, budget, RNG, history and next sweep",
        "phases": ["initialized", "gauss_seidel_sweep_commit", "jacobi_sweep_commit", "paused"],
        "notes": "Compact model-bank codec; mid-sweep child checkpoints are not claimed in this codec.",
    },
    "laboratory.distributed_ls_rqaoa_cpu": {
        "supports_checkpoint": True,
        "boundaries": ["initialized", "reduction_committed"],
        "state": "reduction state, parent means, budget, completed episodes and history",
        "phases": ["initialized", "reduction_committed"],
        "notes": "Outer nested checkpoint codec; mid-episode child resume is not claimed.",
    },
    "laboratory.qaoa_mi_cpu": {
        "supports_checkpoint": True,
        "boundaries": ["completed ADAM iteration", "training_done", "readout_done"],
        "state": "baseline QAOACPU state; MI is recomputed passively from selected final angles",
        "phases": ["training", "training_done", "readout_done"],
        "notes": "Observer diagnostics are not a second RNG/budget; partial MI analysis is not checkpointed.",
    },
    "laboratory.incremental_ls_rqaoa_cpu": {
        "supports_checkpoint": False,
        "boundaries": [],
        "state": "not implemented in a17",
        "phases": [],
        "notes": "Epoch bank and plan are checkpointable in memory records, but no durable codec is declared.",
    },
    "laboratory.rqaoa_mi_cpu": {
        "supports_checkpoint": False,
        "boundaries": [],
        "state": "not implemented",
        "phases": [],
        "notes": "RQAOA recursive checkpoint is not available; MI frames are passive.",
    },
    "qaoa_gpu": {
        "supports_checkpoint": False,
        "boundaries": [],
        "state": "not implemented",
        "phases": [],
        "notes": "A portable CuPy RNG sampler protocol is required first; no pointer snapshot is accepted.",
    },
    "rqaoa_cpu": {
        "supports_checkpoint": False,
        "boundaries": [],
        "state": "not implemented",
        "phases": [],
        "notes": "The a10 capability is declared but the reduction/source codec is not included.",
    },
    "rqaoa_gpu": {
        "supports_checkpoint": False,
        "boundaries": [],
        "state": "not implemented",
        "phases": [],
        "notes": "GPU source and reduction codec are not included.",
    },
    "ls_rqaoa_cpu": {
        "supports_checkpoint": False,
        "boundaries": [],
        "state": "not implemented",
        "phases": [],
        "notes": "Parent/child task barriers need a task codec; not claimed in a10.",
    },
    "ls_rqaoa_gpu": {
        "supports_checkpoint": False,
        "boundaries": [],
        "state": "not implemented",
        "phases": [],
        "notes": "GPU owner and task codec are not included.",
    },
    "qaoa_in_qaoa_cpu": {
        "supports_checkpoint": False,
        "boundaries": [],
        "state": "not implemented",
        "phases": [],
        "notes": "Atomic level projection and local-pattern codec are not included.",
    },
    "qaoa_in_qaoa_gpu": {
        "supports_checkpoint": False,
        "boundaries": [],
        "state": "not implemented",
        "phases": [],
        "notes": "GPU hierarchy codec is not included.",
    },
    "exact_cpu": {
        "supports_checkpoint": False,
        "boundaries": [],
        "state": "not implemented",
        "phases": [],
        "notes": "Batch-level exact enumeration is not claimed in a10.",
    },
    "gw_cpu": {
        "supports_checkpoint": False,
        "boundaries": [],
        "state": "not implemented",
        "phases": [],
        "notes": "CVXPY/SCS is treated as an opaque call; no internal-iteration checkpoint is claimed.",
    },
    "simulated_annealing_cpu": {
        "supports_checkpoint": False,
        "boundaries": [],
        "state": "not implemented",
        "phases": [],
        "notes": "Native SA reads are opaque; no Metropolis-sweep resume is claimed.",
    },
    "refined": {
        "supports_checkpoint": False,
        "boundaries": [],
        "state": "not implemented",
        "phases": [],
        "notes": "Base/refinement child codecs are not included.",
    },
    "exact_block": {
        "supports_checkpoint": False,
        "boundaries": [],
        "state": "not implemented",
        "phases": [],
        "notes": "Partial enumeration is not claimed.",
    },
}


def capability(component_id):
    """Return the declared boundary for one public component."""
    return dict(CAPABILITIES.get(str(component_id), {
        "supports_checkpoint": False,
        "boundaries": [],
        "state": "unknown component",
        "phases": [],
        "notes": "Unknown component has no registered checkpoint codec.",
    }))


def capabilities():
    """Return an independent copy of the full capability table."""
    return {name: dict(value) for name, value in CAPABILITIES.items()}
