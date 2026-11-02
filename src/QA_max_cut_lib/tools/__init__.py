"""Inspect completed runs without rerunning a solver."""

from .numerics import correlation_matrix, quality_score
from .mi_plots import plot_mutual_information_map, plot_mutual_information_frames, plot_mutual_information_distance
from .mutual_information import (
    MutualInformationError, classical_from_counts, classical_from_moments,
    classical_from_probabilities, classical_mutual_information, hop_distances,
    map_summary, pair_scope, quantum_mutual_information, reduced_density_matrix,
    validate_node_order, validate_statevector, von_neumann_entropy,
    partial_trace, entropy, quantum_mi, classical_mi,
)
from ..core.records import load_result
from ..core.saving import save_result
from ..core.factory import load_json

__all__ = [
    "load_json",
    "load_result",
    "save_result",
    "correlation_matrix",
    "quality_score",
    "plot_graph",
    "plot_graph_solution",
    "plot_graph_info",
    "plot_probabilities",
    "plot_optimization",
    "scan_landscape",
    "plot_landscape",
    "plot_reduction_weights",
    "plot_reduction_energy",
    "plot_analytic_gamma_profile",
    "plot_ls_coverage",
    "plot_parameter_trajectories",
    "plot_parameter_trajectories_3d",
    "plot_hierarchy_overview",
    "plot_hierarchy_parameters",
    "plot_hierarchy_parameters_3d",
    "plot_hierarchy_tree",
    "analyze_solution",
    "plot_gw_convergence",
    "plot_sa_reads",
    "plot_refinement",
    "plot_classical_comparison",
    "plan_experiments",
    "run_experiments",
    "ExperimentRunner",
    "load_saved_series",
    "load_saved_records",
    "parallelism_profile",
    "build_profile_plan",
    "plot_series_coverage",
    "plot_series_quality",
    "plot_parallelism",
    "parallelism_html",
    "fit_amdahl",
    "karp_flatt",
    "recommend_total_curve",
    "inspect_checkpoint",
    "restore_checkpoint",
    "TrajectoryPanelCollection",
    "MutualInformationError",
    "validate_statevector",
    "validate_node_order",
    "reduced_density_matrix",
    "von_neumann_entropy",
    "quantum_mutual_information",
    "classical_mutual_information",
    "classical_from_probabilities",
    "classical_from_counts",
    "classical_from_moments",
    "pair_scope",
    "map_summary",
    "hop_distances",
    "partial_trace",
    "entropy",
    "quantum_mi",
    "classical_mi",
    "plot_mutual_information_map",
    "plot_mutual_information_frames",
    "plot_mutual_information_distance",
]


def __getattr__(name):
    if name in {
        "TrajectoryPanelCollection",
    "MutualInformationError",
    "validate_statevector",
    "validate_node_order",
    "reduced_density_matrix",
    "von_neumann_entropy",
    "quantum_mutual_information",
    "classical_mutual_information",
    "classical_from_probabilities",
    "classical_from_counts",
    "classical_from_moments",
    "pair_scope",
    "map_summary",
    "hop_distances",
    "partial_trace",
    "entropy",
    "quantum_mi",
    "classical_mi",
    "plot_mutual_information_map",
    "plot_mutual_information_frames",
    "plot_mutual_information_distance",
        "plot_reduction_weights",
        "plot_reduction_energy",
        "plot_analytic_gamma_profile",
        "plot_ls_coverage",
        "plot_parameter_trajectories",
        "plot_parameter_trajectories_3d",
        "plot_hierarchy_overview",
        "plot_hierarchy_parameters",
        "plot_hierarchy_parameters_3d",
        "plot_hierarchy_tree",
    "analyze_solution",
    "plot_gw_convergence",
    "plot_sa_reads",
    "plot_refinement",
    "plot_classical_comparison",
    "plan_experiments",
    "run_experiments",
    "ExperimentRunner",
    "load_saved_series",
    "load_saved_records",
    "parallelism_profile",
    "build_profile_plan",
    "plot_series_coverage",
    "plot_series_quality",
    "plot_parallelism",
    "parallelism_html",
    "fit_amdahl",
    "karp_flatt",
    "recommend_total_curve",
    "inspect_checkpoint",
    "restore_checkpoint",
    }:
        if name in {
            "plan_experiments",
            "run_experiments",
            "ExperimentRunner",
        }:
            from . import experiments
            return getattr(experiments, name)
        if name in {"load_saved_series", "load_saved_records"}:
            from ..core.series_store import load_saved_records, load_saved_series
            return load_saved_series if name == "load_saved_series" else load_saved_records
        if name in {"inspect_checkpoint", "restore_checkpoint"}:
            from . import checkpoints
            return getattr(checkpoints, name)
        if name in {
            "parallelism_profile",
            "build_profile_plan",
            "fit_amdahl",
            "karp_flatt",
            "recommend_total_curve",
        }:
            from . import parallelism
            return getattr(parallelism, name)
        if name in {
            "plot_series_coverage",
            "plot_series_quality",
            "plot_parallelism",
            "parallelism_html",
        }:
            from . import experiment_plots
            return getattr(experiment_plots, name)
        if name.startswith("plot_hierarchy_"):
            from . import hierarchy_plots

            return getattr(hierarchy_plots, name)
        if name in {
            "analyze_solution",
            "plot_gw_convergence",
            "plot_sa_reads",
            "plot_refinement",
            "plot_classical_comparison",
        }:
            from . import classical_plots

            return getattr(classical_plots, name)
        from . import reduction_plots

        return getattr(reduction_plots, name)
    if name.startswith("plot_") and name in __all__:
        from . import plotting

        return getattr(plotting, name)
    if name == "scan_landscape":
        from .landscape import scan_landscape

        return scan_landscape
    raise AttributeError(name)
