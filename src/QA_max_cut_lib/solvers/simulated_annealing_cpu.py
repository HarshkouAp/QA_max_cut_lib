"""Classical simulated annealing on a spin BQM."""

from __future__ import annotations

from copy import deepcopy
import math
from time import perf_counter

import numpy as np

from .base import BaseSolver, positive_int

__all__ = ["SimulatedAnnealingCPU"]


def _read_seed(root_seed, read_id):
    base = 0 if root_seed is None else int(root_seed)
    sequence = np.random.SeedSequence([base & 0xFFFFFFFF, 2, int(read_id)])
    # The installed dwave-samplers build accepts at most 2**31 - 1 here.
    return int(sequence.generate_state(1, dtype=np.uint32)[0] % (2**31 - 1))


def _one_read(bqm, settings, seed):
    from dwave.samplers import SimulatedAnnealingSampler

    kwargs = dict(
        num_reads=1,
        num_sweeps=settings["num_sweeps"],
        beta_schedule_type=settings["beta_schedule_type"],
        seed=seed,
    )
    if settings["beta_range"] is not None:
        kwargs["beta_range"] = settings["beta_range"]
    sample_set = SimulatedAnnealingSampler().sample(bqm, **kwargs)
    return sample_set


class SimulatedAnnealingCPU(BaseSolver):
    """Run independent classical simulated-annealing reads.

    Parameters
    ----------
    config : dict, optional
        ``parameters`` sets ``num_reads``, ``num_sweeps``, schedule and
        optional ``beta_range``.  ``limits`` accepts ``anneal_reads_total``
        and ``anneal_sweeps_total``.  ``collect.candidate_history`` stores
        one compact record per logical read.

    Notes
    -----
    One sweep is a full sequential spin-update pass.  One read is one
    independent annealing run.  These are not quantum shots.  The all-plus
    assignment is an incumbent, not a hidden initial state for every read.
    The returned sample is the best final read state plus that incumbent.
    """

    algorithm_type = "simulated_annealing_cpu"
    supports_parallel = True
    optional_fields = BaseSolver.optional_fields + (
        "num_reads_requested",
        "num_reads_started",
        "num_reads_completed",
        "num_sweeps_requested",
        "num_sweeps_accounted",
        "beta_schedule_type",
        "beta_range",
        "effective_beta_range",
        "bqm_offset",
        "chosen_origin",
        "seed_plan",
        "candidate_history",
        "solution_source",
    )

    def _validate(self, config, graph):
        params = config.setdefault("parameters", {})
        allowed = {
            "num_reads",
            "num_sweeps",
            "beta_schedule_type",
            "beta_range",
        }
        if set(params) - allowed:
            raise ValueError("Unknown simulated-annealing parameter.")
        params.setdefault("num_reads", 10)
        params.setdefault("num_sweeps", 64)
        params.setdefault("beta_schedule_type", "linear")
        params.setdefault("beta_range", None)
        positive_int(params["num_reads"], "num_reads")
        positive_int(params["num_sweeps"], "num_sweeps")
        if params["beta_schedule_type"] not in ("linear", "geometric"):
            raise ValueError("beta_schedule_type must be linear or geometric.")
        beta_range = params["beta_range"]
        if beta_range is not None:
            if (
                not isinstance(beta_range, (list, tuple))
                or len(beta_range) != 2
                or not all(np.isfinite(value) for value in beta_range)
                or not float(beta_range[0]) < float(beta_range[1])
            ):
                raise ValueError("beta_range must be a finite increasing pair.")
            if params["beta_schedule_type"] == "geometric" and float(beta_range[0]) <= 0:
                raise ValueError("geometric beta_range must be strictly positive.")
            params["beta_range"] = [float(beta_range[0]), float(beta_range[1])]
        limits = config.setdefault("limits", {})
        if set(limits) - {"anneal_reads_total", "anneal_sweeps_total"}:
            raise ValueError("Unknown simulated-annealing limit.")
        limits.setdefault("anneal_reads_total", None)
        limits.setdefault("anneal_sweeps_total", None)
        for key in ("anneal_reads_total", "anneal_sweeps_total"):
            if limits[key] is not None:
                positive_int(limits[key], key, minimum=0)
        collect = config.setdefault("collect", {})
        if set(collect) - {"candidate_history"}:
            raise ValueError("Unknown simulated-annealing collect field.")
        collect.setdefault("candidate_history", False)
        if type(collect["candidate_history"]) is not bool:
            raise ValueError("collect.candidate_history must be boolean.")

    def _run(self, graph, config, rng):
        from ..tools.reporting import log_message

        try:
            import dimod
        except ImportError as error:  # pragma: no cover - dependency-specific
            raise ImportError(
                "Simulated annealing requires dimod and dwave-samplers. "
                "Install the project's classical extra."
            ) from error
        nodes = tuple(graph.nodes)
        h = {node: 0.0 for node in nodes}
        J = {(u, v): 0.5 * float(weight) for u, v, weight in graph.edges}
        offset = -0.5 * math.fsum(float(weight) for _, _, weight in graph.edges)
        bqm = dimod.BinaryQuadraticModel.from_ising(h, J, offset=offset)
        params = config["parameters"]
        limits = config["limits"]
        requested_reads = int(params["num_reads"])
        num_sweeps = int(params["num_sweeps"])
        reads_limit = limits["anneal_reads_total"]
        sweeps_limit = limits["anneal_sweeps_total"]
        allowed_reads = requested_reads
        reason = "completed"
        if reads_limit is not None:
            allowed_reads = min(allowed_reads, int(reads_limit))
            if allowed_reads < requested_reads:
                reason = "anneal_reads_total"
        if sweeps_limit is not None:
            sweep_reads = int(sweeps_limit) // num_sweeps
            if sweep_reads < allowed_reads:
                allowed_reads = sweep_reads
                reason = "anneal_sweeps_total"
        if allowed_reads <= 0:
            reason = reason if reason != "completed" else "anneal_reads_total"
        status = "completed" if allowed_reads == requested_reads else "stopped"
        root_seed = config.get("seed")
        if root_seed is None:
            root_seed = int(rng.integers(0, 2**32, dtype=np.uint32))
        seed_plan = [_read_seed(root_seed, index) for index in range(allowed_reads)]
        history = [] if config["collect"]["candidate_history"] else None
        best = {node: 1 for node in nodes}
        best_cut = 0.0
        chosen_origin = "all_plus_incumbent"
        reads_completed = 0
        sweeps_accounted = 0
        effective_beta_range = None
        started = perf_counter()
        settings = {
            "num_sweeps": num_sweeps,
            "beta_schedule_type": params["beta_schedule_type"],
            "beta_range": params["beta_range"],
        }

        def run_one(read_id):
            return read_id, _one_read(bqm, settings, seed_plan[read_id])

        results = []
        if allowed_reads:
            if int(config["num_treads"]) == 1:
                results = [run_one(index) for index in range(allowed_reads)]
            else:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=int(config["num_treads"])) as pool:
                    results = list(pool.map(run_one, range(allowed_reads)))
        # Deterministic merge by logical read_id, never completion order.
        results.sort(key=lambda item: item[0])
        for read_id, sample_set in results:
            if sample_set.vartype != dimod.SPIN:
                raise RuntimeError("The annealing sampler did not return spin states.")
            variables = list(sample_set.variables)
            if set(variables) != set(nodes):
                raise RuntimeError("The annealing sample does not contain every graph node.")
            record = sample_set.record
            if len(record.sample) != 1:
                raise RuntimeError("Each logical read must return exactly one sample.")
            occurrence = int(np.sum(record.num_occurrences))
            sample = {
                node: int(record.sample[0][variables.index(node)]) for node in nodes
            }
            if any(value not in (-1, 1) for value in sample.values()):
                raise RuntimeError("The annealing sample contains a non-spin value.")
            cut = graph.cut_value(sample)
            energy = -cut
            bqm_energy = float(bqm.energy(sample))
            if not math.isclose(energy, bqm_energy, rel_tol=1e-9, abs_tol=1e-9 * max(1.0, abs(energy))):
                raise ArithmeticError("The annealing BQM energy and direct cut disagree.")
            selected = cut > best_cut
            if selected:
                best = sample
                best_cut = cut
                chosen_origin = f"read:{read_id}"
            reads_completed += 1
            sweeps_accounted += num_sweeps
            info = sample_set.info or {}
            if effective_beta_range is None and info.get("beta_range") is not None:
                effective_beta_range = list(info["beta_range"])
            if history is not None:
                history.append(
                    {
                        "read_id": read_id,
                        "seed": seed_plan[read_id],
                        "cut_value": float(cut),
                        "solution_energy": float(energy),
                        "bqm_energy": bqm_energy,
                        "running_best": float(best_cut),
                        "selected": bool(selected),
                        "num_occurrences": occurrence,
                        "seconds": float(getattr(sample_set, "info", {}).get("timing", {}).get("qpu_sampling_time", 0.0) or 0.0),
                    }
                )
        log_message(
            config,
            f"Simulated annealing completed {reads_completed}/{requested_reads} reads.",
            detail=False,
        )
        return {
            "solution": best,
            "status": status,
            "termination_reason": reason,
            "solution_source": chosen_origin,
            "num_reads_requested": requested_reads,
            "num_reads_started": len(results),
            "num_reads_completed": reads_completed,
            "num_sweeps_requested": requested_reads * num_sweeps,
            "num_sweeps_accounted": sweeps_accounted,
            "beta_schedule_type": params["beta_schedule_type"],
            "beta_range": deepcopy(params["beta_range"]),
            "effective_beta_range": effective_beta_range,
            "bqm_offset": float(offset),
            "chosen_origin": chosen_origin,
            "seed_plan": seed_plan,
            "candidate_history": history,
            "resources": {
                "shots_used": 0,
                "anneal_reads_requested": requested_reads,
                "anneal_reads_started": len(results),
                "anneal_reads_completed": reads_completed,
                "anneal_sweeps_accounted": sweeps_accounted,
                "anneal_sweeps_upper_bound": reads_completed * num_sweeps,
                "num_treads": int(config["num_treads"]),
                "total_seconds": perf_counter() - started,
            },
            "data_status": {
                "candidate_history": "collected" if history is not None else "not_collected"
            },
        }


def main(argv=None):
    """Run one simulated-annealing example through the common interface."""
    from ..tools.demo import run_solver_demo

    return run_solver_demo(SimulatedAnnealingCPU, argv)


if __name__ == "__main__":
    raise SystemExit(main())
