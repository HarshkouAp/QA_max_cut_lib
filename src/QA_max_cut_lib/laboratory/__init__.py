"""Laboratory namespace for reviewed historical variants.

All catalogued recipes have terminal dispositions; every card is runnable
through a shared CPU controller or a legacy compatibility adapter.
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
IMPLEMENTED_VARIANTS = tuple(sorted({
    "warm_start_product_qaoa_cpu_v1", "warm_start_ensemble_rqaoa_cpu_v1",
    "conditional_fields_cpu_v1", "block_mean_field_gauss_seidel_cpu_v1",
    "block_mean_field_jacobi_cpu_v1", "distributed_ls_rqaoa_carry_cpu_v1",
    "distributed_ls_rqaoa_cold_cpu_v1", "qaoa_mi_observer_cpu_v1",
    "rqaoa_mi_observer_cpu_v1", "incremental_ls_mk_components_v1",
    "legacy_ls_rqaoa_fast_alias", "warm_start_rqaoa", "laboratory_ls_families",
    "mutual_information_diagnostics", "deferred_environment_and_generators",
}))


def __getattr__(name):
    if name in {
        "LSRQAOAExp", "LSRQAOAFast1", "LSRQAOAFast2", "LSRQAOAFast3",
        "LSRQAOAFast4", "MK6V1", "MK6V2", "MK7V1", "MK7V2",
        "QAOAAdam", "QAOAF2", "QAOAFM", "CorrectedMeanMap",
        "CorrelationExtractor", "EdgeExtender", "GraphDecomposerLight",
        "GraphScanner", "partition_smoke_test",
    }:
        from . import legacy
        value = getattr(legacy, name)
        globals()[name] = value
        return value
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
    "LSRQAOAExp",
    "LSRQAOAFast1",
    "LSRQAOAFast2",
    "LSRQAOAFast3",
    "LSRQAOAFast4",
    "MK6V1",
    "MK6V2",
    "MK7V1",
    "MK7V2",
    "QAOAAdam",
    "QAOAF2",
    "QAOAFM",
    "CorrectedMeanMap",
    "CorrelationExtractor",
    "EdgeExtender",
    "GraphDecomposerLight",
    "GraphScanner",
    "partition_smoke_test",
    "IMPLEMENTED_VARIANTS",
    "catalog_summary",
    "get_file",
    "get_variant",
    "list_variants",
    "load_catalog",
    "load_inventory",
    "source_to_variant",
]

