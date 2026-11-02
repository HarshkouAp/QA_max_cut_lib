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
        return ("exact_cpu", "qaoa_cpu", "rqaoa_cpu")

    @staticmethod
    def from_config(config):
        """Return the selected implemented solver. Never change backend."""
        from ..solvers.exact_cpu import ExactCPU
        from ..solvers.qaoa_cpu import QAOACPU
        from ..solvers.rqaoa_cpu import RQAOACPU

        types = {
            "exact_cpu": ExactCPU,
            "qaoa_cpu": QAOACPU,
            "rqaoa_cpu": RQAOACPU,
        }
        if config.get("type") not in types:
            raise ValueError(
                f"Solver {config.get('type')!r} is not implemented in {__version__}. Available: {tuple(types)}"
            )
        return types[config["type"]](config)
