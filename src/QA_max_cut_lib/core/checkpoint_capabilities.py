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
