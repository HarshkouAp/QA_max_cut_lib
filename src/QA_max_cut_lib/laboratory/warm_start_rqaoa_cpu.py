"""Public Laboratory module for multi-start warm-start RQAOA."""

from ..solvers.warm_start_rqaoa_cpu import WarmStartRQAOACPU, main

__all__ = ["WarmStartRQAOACPU", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
