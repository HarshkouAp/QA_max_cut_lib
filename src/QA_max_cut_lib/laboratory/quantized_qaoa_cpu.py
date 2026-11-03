"""Public route for quantized-angle QAOA (CPU)."""
from ..solvers.quantized_qaoa import QuantizedQAOACPU

def main(argv=None):
    from ..tools.demo import run_solver_demo
    return run_solver_demo(QuantizedQAOACPU, argv)

__all__ = ["QuantizedQAOACPU", "main"]
if __name__ == "__main__":
    raise SystemExit(main())
