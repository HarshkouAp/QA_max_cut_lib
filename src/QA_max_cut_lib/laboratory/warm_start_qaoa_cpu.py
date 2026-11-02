"""Public Laboratory module for the first warm-start CPU solver.

The implementation lives in ``solvers.warm_start_qaoa_cpu`` so it shares the
existing QAOA lifecycle.  This module is the stable public import/CLI route.
"""

from ..solvers.warm_start_qaoa_cpu import WarmStartQAOACPU, main

__all__ = ["WarmStartQAOACPU", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
