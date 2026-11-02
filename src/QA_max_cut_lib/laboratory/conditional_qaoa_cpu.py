"""Public Laboratory module for fixed-spin conditional QAOA."""

from ..solvers.conditional_qaoa_cpu import ConditionalQAOACPU, main

__all__ = ["ConditionalQAOACPU", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
