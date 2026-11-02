"""Goemans-Williamson SDP relaxation with randomized hyperplane rounding."""

from __future__ import annotations

from copy import deepcopy
import math
from time import perf_counter

import numpy as np

from .base import BaseSolver, positive_int

__all__ = ["GoemansWilliamsonCPU"]


class GoemansWilliamsonCPU(BaseSolver):
    """Solve a weighted Max-Cut by an SCS SDP relaxation and GW rounding.

    Parameters
    ----------
    config : dict, optional
        ``parameters`` contains ``num_hyperplanes``, SCS tolerances and
        matrix-repair tolerances.  ``limits`` accepts ``sdp_time_limit``.
        ``collect`` controls optional dense matrix/factor storage.

    Attributes
    ----------
    solution : dict of int to int
        Best complete rounded assignment.  The all-plus incumbent is always
        present and has cut zero.
    sdp_status : str
        CVXPY/SCS status.
    sdp_objective_raw, sdp_objective_repaired : float
        Relaxation values before and after numerical PSD repair.
    sdp_upper_bound : None
        No strict dual certificate is built in this milestone.
    positive_weight_upper_bound : float
        Weak analytic bound ``sum(max(w, 0))`` with an explicit kind tag.
    candidate_history : list of dict or None
        One record per hyperplane when collected.

    Notes
    -----
    The objective and relaxation are exact mathematical statements.  The
    numerical SDP value is not a certified upper bound after approximate
    repair.  The classical 0.878 approximation statement is not used as a
    per-run guarantee for signed weights.
    """

    algorithm_type = "gw_cpu"
    optional_fields = BaseSolver.optional_fields + (
        "sdp_status",
        "sdp_objective_raw",
        "sdp_objective_repaired",
        "sdp_upper_bound",
        "sdp_upper_bound_reason",
        "positive_weight_upper_bound",
        "bound_kind",
        "num_hyperplanes_requested",
        "num_hyperplanes_completed",
        "chosen_origin",
        "solver_options",
        "sdp_diagnostics",
        "candidate_history",
        "sdp_matrix",
        "sdp_factor",
        "solution_source",
    )

    def _validate(self, config, graph):
        params = config.setdefault("parameters", {})
        allowed = {
            "num_hyperplanes",
            "solver",
            "eps_abs",
            "eps_rel",
            "max_iters",
            "allow_inaccurate",
            "psd_tolerance",
            "diag_tolerance",
            "repair_tolerance",
            "time_limit_secs",
        }
        if set(params) - allowed:
            raise ValueError("Unknown GW parameter.")
        params.setdefault("num_hyperplanes", 10)
        params.setdefault("solver", "SCS")
        params.setdefault("eps_abs", 1e-6)
        params.setdefault("eps_rel", 1e-6)
        params.setdefault("max_iters", 10000)
        params.setdefault("allow_inaccurate", False)
        params.setdefault("psd_tolerance", 1e-6)
        params.setdefault("diag_tolerance", 1e-6)
        params.setdefault("repair_tolerance", 1e-6)
        params.setdefault("time_limit_secs", None)
        if params["solver"] != "SCS":
            raise ValueError("The first GW implementation supports solver='SCS' only.")
        positive_int(params["num_hyperplanes"], "num_hyperplanes")
        for key in ("eps_abs", "eps_rel", "psd_tolerance", "diag_tolerance", "repair_tolerance"):
            value = params[key]
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive.")
        positive_int(params["max_iters"], "max_iters")
        if type(params["allow_inaccurate"]) is not bool:
            raise ValueError("allow_inaccurate must be boolean.")
        if params["time_limit_secs"] is not None:
            value = params["time_limit_secs"]
            if not np.isfinite(value) or value <= 0:
                raise ValueError("time_limit_secs must be finite and positive.")
        limits = config.setdefault("limits", {})
        if set(limits) - {"sdp_time_limit"}:
            raise ValueError("Unknown GW limit.")
        limits.setdefault("sdp_time_limit", None)
        if limits["sdp_time_limit"] is not None:
            value = limits["sdp_time_limit"]
            if not np.isfinite(value) or value <= 0:
                raise ValueError("sdp_time_limit must be finite and positive.")
        collect = config.setdefault("collect", {})
        allowed_collect = {"candidate_history", "sdp_matrix", "sdp_factor"}
        if set(collect) - allowed_collect:
            raise ValueError("Unknown GW collect field.")
        for key in allowed_collect:
            collect.setdefault(key, False)
            if type(collect[key]) is not bool:
                raise ValueError(f"collect.{key} must be boolean.")
        if len(graph) > 0 and (
            not isinstance(params["num_hyperplanes"], int)
            or params["num_hyperplanes"] > 1_000_000
        ):
            raise ValueError("num_hyperplanes is unreasonably large.")

    def _run(self, graph, config, rng):
        from ..tools.reporting import log_message

        params = config["parameters"]
        nodes = tuple(graph.nodes)
        n = len(nodes)
        started = perf_counter()
        all_plus = {node: 1 for node in nodes}
        best = all_plus
        best_cut = 0.0
        chosen_origin = "all_plus_incumbent"
        candidates = [] if config["collect"]["candidate_history"] else None
        sdp_matrix = None
        sdp_factor = None
        if n == 0:
            return {
                "solution": {},
                "status": "completed",
                "termination_reason": "empty_graph",
                "solution_source": "empty_graph",
                "sdp_status": "not_applicable",
                "sdp_objective_raw": 0.0,
                "sdp_objective_repaired": 0.0,
                "sdp_upper_bound": None,
                "sdp_upper_bound_reason": "empty_graph",
                "positive_weight_upper_bound": 0.0,
                "bound_kind": "sum_positive_weights",
                "num_hyperplanes_requested": int(params["num_hyperplanes"]),
                "num_hyperplanes_completed": 0,
                "chosen_origin": "all_plus_incumbent",
                "solver_options": {},
                "candidate_history": candidates,
                "sdp_matrix": None,
                "sdp_factor": None,
                "resources": {
                    "shots_used": 0,
                    "sdp_calls": 0,
                    "hyperplanes": 0,
                    "num_treads": 1,
                    "solve_seconds": perf_counter() - started,
                },
                "data_status": {"sdp_matrix": "not_applicable", "sdp_factor": "not_applicable"},
            }
        try:
            import cvxpy as cp
        except ImportError as error:  # pragma: no cover - dependency-specific
            raise ImportError(
                "GW requires cvxpy and scs. Install the project's classical extra."
            ) from error
        X = cp.Variable((n, n), symmetric=True)
        edge_terms = []
        weights = []
        for u, v, weight in graph.edges:
            weights.append(float(weight))
            i = nodes.index(u)
            j = nodes.index(v)
            edge_terms.append(float(weight) * X[i, j])
        if edge_terms:
            objective = cp.Maximize(
                0.5 * math.fsum(weights) - 0.5 * sum(edge_terms)
            )
        else:
            objective = cp.Maximize(0.0)
        constraints = [X >> 0] + [X[i, i] == 1 for i in range(n)]
        problem = cp.Problem(objective, constraints)
        solver_kwargs = dict(
            solver="SCS",
            warm_start=False,
            eps_abs=float(params["eps_abs"]),
            eps_rel=float(params["eps_rel"]),
            max_iters=int(params["max_iters"]),
            verbose=False,
        )
        time_limit = params["time_limit_secs"]
        if time_limit is None:
            time_limit = config["limits"]["sdp_time_limit"]
        if time_limit is not None:
            solver_kwargs["time_limit_secs"] = float(time_limit)
        solve_started = perf_counter()
        problem.solve(**solver_kwargs)
        solve_seconds = perf_counter() - solve_started
        status = str(problem.status)
        if status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"GW SDP did not solve successfully: {status}.")
        if status == "optimal_inaccurate" and not params["allow_inaccurate"]:
            raise RuntimeError(
                "GW SDP returned optimal_inaccurate; set allow_inaccurate=true explicitly."
            )
        raw = float(problem.value)
        if not np.isfinite(raw) or X.value is None:
            raise RuntimeError("GW SDP returned a non-finite value or no matrix.")
        matrix = np.asarray(X.value, dtype=np.float64)
        if matrix.shape != (n, n) or not np.isfinite(matrix).all():
            raise RuntimeError("GW SDP matrix is not finite or has the wrong shape.")
        matrix = 0.5 * (matrix + matrix.T)
        diagonal_residual = float(np.max(np.abs(np.diag(matrix) - 1.0)))
        if diagonal_residual > float(params["diag_tolerance"]):
            raise RuntimeError(
                f"GW SDP diagonal residual {diagonal_residual} exceeds tolerance."
            )
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        minimum_eigenvalue = float(eigenvalues[0])
        psd_tol = float(params["psd_tolerance"])
        scale = max(1.0, float(np.linalg.norm(matrix, ord=2)))
        if minimum_eigenvalue < -psd_tol * scale:
            raise RuntimeError(
                f"GW SDP matrix is not PSD: min eigenvalue {minimum_eigenvalue}."
            )
        repaired = np.maximum(eigenvalues, 0.0)
        repaired_matrix = (eigenvectors * repaired) @ eigenvectors.T
        repair_norm = float(np.linalg.norm(repaired_matrix - matrix, ord="fro"))
        if repair_norm > float(params["repair_tolerance"]) * max(1.0, scale):
            # Small spectral clipping is allowed; a large unexplained change is not.
            if minimum_eigenvalue < 0:
                raise RuntimeError(
                    f"GW PSD repair is too large: {repair_norm}."
                )
        factor = eigenvectors * np.sqrt(repaired)
        norms = np.linalg.norm(factor, axis=1)
        if not np.isfinite(norms).all() or np.min(norms) <= float(params["repair_tolerance"]):
            raise RuntimeError("GW factor contains a zero or non-finite row norm.")
        factor = factor / norms[:, None]
        repaired_value = 0.5 * math.fsum(
            float(weight) * (1.0 - float(factor[nodes.index(u)] @ factor[nodes.index(v)]))
            for u, v, weight in graph.edges
        )
        if config["collect"]["sdp_matrix"]:
            sdp_matrix = matrix.copy()
        if config["collect"]["sdp_factor"]:
            sdp_factor = factor.copy()
        positive_bound = float(
            math.fsum(max(float(weight), 0.0) for _, _, weight in graph.edges)
        )
        for trial in range(int(params["num_hyperplanes"])):
            normal = rng.normal(size=n)
            projection = factor @ normal
            spins = {}
            for index, node in enumerate(nodes):
                spins[node] = 1 if projection[index] >= 0.0 else -1
            candidate_cut = graph.cut_value(spins)
            origin = f"hyperplane:{trial}"
            selected = candidate_cut > best_cut
            if selected:
                best = spins
                best_cut = candidate_cut
                chosen_origin = origin
            if candidates is not None:
                candidates.append(
                    {
                        "trial_id": trial,
                        "origin": origin,
                        "cut_value": float(candidate_cut),
                        "running_best": float(best_cut),
                        "selected": bool(selected),
                        "num_positive_projection": int(np.count_nonzero(projection >= 0.0)),
                    }
                )
        log_message(config, "GW SDP and hyperplane rounding completed.", detail=False)
        return {
            "solution": best,
            "status": "completed",
            "termination_reason": "completed",
            "solution_source": chosen_origin,
            "sdp_status": status,
            "sdp_objective_raw": raw,
            "sdp_objective_repaired": repaired_value,
            "sdp_upper_bound": None,
            "sdp_upper_bound_reason": "No strict dual certificate is built in this milestone.",
            "positive_weight_upper_bound": positive_bound,
            "bound_kind": "sum_positive_weights",
            "num_hyperplanes_requested": int(params["num_hyperplanes"]),
            "num_hyperplanes_completed": int(params["num_hyperplanes"]),
            "chosen_origin": chosen_origin,
            "solver_options": {
                "solver": "SCS",
                "warm_start": False,
                "eps_abs": float(params["eps_abs"]),
                "eps_rel": float(params["eps_rel"]),
                "max_iters": int(params["max_iters"]),
                "allow_inaccurate": bool(params["allow_inaccurate"]),
            },
            "sdp_diagnostics": {
                "minimum_eigenvalue": minimum_eigenvalue,
                "diagonal_residual": diagonal_residual,
                "repair_norm": repair_norm,
                "row_norm_min": float(np.min(norms)),
                "row_norm_max": float(np.max(norms)),
            },
            "candidate_history": candidates,
            "sdp_matrix": sdp_matrix,
            "sdp_factor": sdp_factor,
            "resources": {
                "shots_used": 0,
                "sdp_calls": 1,
                "hyperplanes": int(params["num_hyperplanes"]),
                "num_treads": 1,
                "solve_seconds": solve_seconds,
                "total_seconds": perf_counter() - started,
            },
            "data_status": {
                "sdp_matrix": "collected" if sdp_matrix is not None else "not_collected",
                "sdp_factor": "collected" if sdp_factor is not None else "not_collected",
            },
        }


def main(argv=None):
    """Run one GW example through the common command interface."""
    from ..tools.demo import run_solver_demo

    return run_solver_demo(GoemansWilliamsonCPU, argv)


if __name__ == "__main__":
    raise SystemExit(main())
