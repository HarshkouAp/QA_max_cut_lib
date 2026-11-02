"""Public Laboratory route for distributed LS-RQAOA."""

from ..solvers.distributed_ls import DistributedLSRQAOACPU, main

__all__ = ["DistributedLSRQAOACPU", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
