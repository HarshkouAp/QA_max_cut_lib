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
from ..solvers.warm_start_qaoa_cpu import WarmStartQAOACPU

IMPLEMENTED_VARIANTS = ("warm_start_product_qaoa_cpu_v1",)

__all__ = [
    "CatalogError",
    "UnknownVariant",
    "UnavailableVariant",
    "WarmStartQAOACPU",
    "IMPLEMENTED_VARIANTS",
    "catalog_summary",
    "get_file",
    "get_variant",
    "list_variants",
    "load_catalog",
    "load_inventory",
    "source_to_variant",
]

