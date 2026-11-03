"""Laboratory namespace for reviewed historical variants.

Only explicitly implemented recipes are runnable.  Catalogued but deferred
cards remain visible and raise ``UnavailableVariant`` if a caller asks the
factory to run them.
"""

from .catalog import (
    CatalogError,
    UnknownVariant,
    UnavailableVariant,
    catalog_summary,
    get_file,
    get_variant,
    list_variants,
    load_catalog,
    load_inventory,
    source_to_variant,
)
IMPLEMENTED_VARIANTS = ("warm_start_product_qaoa_cpu_v1",)


def __getattr__(name):
    if name == "WarmStartQAOACPU":
        from .warm_start_qaoa_cpu import WarmStartQAOACPU
        globals()[name] = WarmStartQAOACPU
        return WarmStartQAOACPU
    if name in ("QAOAWithMIObserverCPU", "RQAOAWithMIObserverCPU"):
        from ..solvers.mi_observer import QAOAWithMIObserverCPU, RQAOAWithMIObserverCPU
        globals()[name] = {"QAOAWithMIObserverCPU": QAOAWithMIObserverCPU, "RQAOAWithMIObserverCPU": RQAOAWithMIObserverCPU}[name]
        return globals()[name]
    if name == "BatchLSRQAOACPU":
        from ..solvers.batch_ls import BatchLSRQAOACPU
        globals()[name] = BatchLSRQAOACPU
        return globals()[name]
    if name == "QuantizedQAOACPU":
        from ..solvers.quantized_qaoa import QuantizedQAOACPU
        globals()[name] = QuantizedQAOACPU
        return globals()[name]
    if name == "IncrementalLSRQAOACPU":
        from ..solvers.incremental_ls import IncrementalLSRQAOACPU
        globals()[name] = IncrementalLSRQAOACPU
        return globals()[name]
    if name == "DistributedLSRQAOACPU":
        from .distributed_ls_rqaoa_cpu import DistributedLSRQAOACPU
        globals()[name] = DistributedLSRQAOACPU
        return DistributedLSRQAOACPU
    if name == "BlockMeanFieldQAOACPU":
        from .block_mean_field_qaoa_cpu import BlockMeanFieldQAOACPU
        globals()[name] = BlockMeanFieldQAOACPU
        return BlockMeanFieldQAOACPU
    if name == "EnvironmentLSRQAOACPU":
        from .environment_ls_rqaoa_cpu import EnvironmentLSRQAOACPU
        globals()[name] = EnvironmentLSRQAOACPU
        return EnvironmentLSRQAOACPU
    if name == "ConditionalQAOACPU":
        from .conditional_qaoa_cpu import ConditionalQAOACPU
        globals()[name] = ConditionalQAOACPU
        return ConditionalQAOACPU
    if name == "WarmStartRQAOACPU":
        from .warm_start_rqaoa_cpu import WarmStartRQAOACPU
        globals()[name] = WarmStartRQAOACPU
        return WarmStartRQAOACPU
    raise AttributeError(name)


__all__ = [
    "CatalogError",
    "UnknownVariant",
    "UnavailableVariant",
    "WarmStartQAOACPU",
    "WarmStartRQAOACPU",
    "ConditionalQAOACPU",
    "EnvironmentLSRQAOACPU",
    "BlockMeanFieldQAOACPU",
    "DistributedLSRQAOACPU",
    "IncrementalLSRQAOACPU",
    "BatchLSRQAOACPU",
    "QuantizedQAOACPU",
    "QAOAWithMIObserverCPU",
    "RQAOAWithMIObserverCPU",
    "IMPLEMENTED_VARIANTS",
    "catalog_summary",
    "get_file",
    "get_variant",
    "list_variants",
    "load_catalog",
    "load_inventory",
    "source_to_variant",
]

