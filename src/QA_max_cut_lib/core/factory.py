"""Build the solvers implemented in this milestone."""

import json
from pathlib import Path

from .._version import __version__


def load_json(path):
    """Read a JSON configuration from a UTF-8 file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


class SolverFactory:
    """Construct a solver from its type and configuration."""

    @staticmethod
    def available():
        return (
            "exact_cpu",
            "qaoa_cpu",
            "rqaoa_cpu",
            "ls_rqaoa_cpu",
            "qaoa_gpu",
            "rqaoa_gpu",
            "ls_rqaoa_gpu",
            "qaoa_in_qaoa_cpu",
            "qaoa_in_qaoa_gpu",
            "gw_cpu",
            "simulated_annealing_cpu",
            "refined",
            "laboratory.warm_start_qaoa_cpu",
        )

    @staticmethod
    def from_config(config):
        """Return the selected implemented solver. Never change backend."""
        from ..solvers.exact_cpu import ExactCPU
        from ..solvers.qaoa_cpu import QAOACPU
        from ..solvers.rqaoa_cpu import RQAOACPU
        from ..solvers.ls_rqaoa_cpu import LSRQAOACPU
        from ..solvers.qaoa_gpu import QAOAGPU
        from ..solvers.rqaoa_gpu import RQAOAGPU
        from ..solvers.ls_rqaoa_gpu import LSRQAOAGPU
        from ..solvers.qaoa_in_qaoa_cpu import QAOAInQAOACPU
        from ..solvers.qaoa_in_qaoa_gpu import QAOAInQAOAGPU
        from ..solvers.goemans_williamson_cpu import GoemansWilliamsonCPU
        from ..solvers.simulated_annealing_cpu import SimulatedAnnealingCPU
        from ..solvers.refined import RefinedSolver
        from ..solvers.warm_start_qaoa_cpu import WarmStartQAOACPU

        types = {
            "exact_cpu": ExactCPU,
            "qaoa_cpu": QAOACPU,
            "rqaoa_cpu": RQAOACPU,
            "ls_rqaoa_cpu": LSRQAOACPU,
            "qaoa_gpu": QAOAGPU,
            "rqaoa_gpu": RQAOAGPU,
            "ls_rqaoa_gpu": LSRQAOAGPU,
            "qaoa_in_qaoa_cpu": QAOAInQAOACPU,
            "qaoa_in_qaoa_gpu": QAOAInQAOAGPU,
            "gw_cpu": GoemansWilliamsonCPU,
            "simulated_annealing_cpu": SimulatedAnnealingCPU,
            "refined": RefinedSolver,
            "laboratory.warm_start_qaoa_cpu": WarmStartQAOACPU,
        }
        if config.get("type") not in types:
            raise ValueError(
                f"Solver {config.get('type')!r} is not implemented in {__version__}. Available: {tuple(types)}"
            )
        return types[config["type"]](config)
