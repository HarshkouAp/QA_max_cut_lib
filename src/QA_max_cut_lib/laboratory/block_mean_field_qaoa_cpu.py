"""Public Laboratory route for block mean-field QAOA."""

from ..solvers.block_mean_field_qaoa_cpu import BlockMeanFieldQAOACPU, main

__all__ = ["BlockMeanFieldQAOACPU", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
