"""Public Laboratory module for environment-aware LS-RQAOA."""

from ..solvers.environment_ls_rqaoa_cpu import EnvironmentLSRQAOACPU, main

__all__ = ["EnvironmentLSRQAOACPU", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
