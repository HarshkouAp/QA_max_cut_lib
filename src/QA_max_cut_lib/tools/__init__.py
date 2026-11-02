"""Inspect completed runs without rerunning a solver."""

from .numerics import correlation_matrix, quality_score
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
    "plot_parameter_trajectories",
    "plot_parameter_trajectories_3d",
    "TrajectoryPanelCollection",
]


def __getattr__(name):
    if name in {
        "TrajectoryPanelCollection",
        "plot_reduction_weights",
        "plot_reduction_energy",
        "plot_analytic_gamma_profile",
        "plot_parameter_trajectories",
        "plot_parameter_trajectories_3d",
    }:
        from . import reduction_plots

        return getattr(reduction_plots, name)
    if name.startswith("plot_") and name in __all__:
        from . import plotting

        return getattr(plotting, name)
    if name == "scan_landscape":
        from .landscape import scan_landscape

        return scan_landscape
    raise AttributeError(name)
