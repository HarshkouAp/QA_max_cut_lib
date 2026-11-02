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
    "IMPLEMENTED_VARIANTS",
    "catalog_summary",
    "get_file",
    "get_variant",
    "list_variants",
    "load_catalog",
    "load_inventory",
    "source_to_variant",
]

