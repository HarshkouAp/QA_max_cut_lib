"""Versioned, import-light access to the historical Laboratory catalog.

The catalog is a package resource.  It contains review/status metadata only;
it does not import archived modules or optional numerical backends.
"""

from __future__ import annotations

from copy import deepcopy
from importlib.resources import files
import json
from typing import Any

__all__ = [
    "CatalogError",
    "UnknownVariant",
    "UnavailableVariant",
    "load_catalog",
    "load_inventory",
    "list_variants",
    "get_variant",
    "get_file",
    "source_to_variant",
    "catalog_summary",
]


class CatalogError(LookupError):
    """Raised when a catalog resource is malformed or inconsistent."""


class UnknownVariant(CatalogError):
    """Raised when a requested variant_id is absent from the catalog."""


class UnavailableVariant(CatalogError):
    """Raised when a catalogued variant is not runnable in this release."""


def _read(name: str) -> dict[str, Any]:
    try:
        text = files("QA_max_cut_lib.laboratory").joinpath("data", name).read_text(
            encoding="utf-8"
        )
    except FileNotFoundError as exc:
        raise CatalogError(f"Laboratory resource is missing: {name}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CatalogError(f"Laboratory resource is not valid JSON: {name}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise CatalogError(f"Unsupported Laboratory resource schema: {name}")
    return data


def load_catalog() -> dict[str, Any]:
    """Return the versioned variant catalog as an independent dictionary."""
    return _read("catalog.json")


def load_inventory() -> dict[str, Any]:
    """Return the versioned source inventory as an independent dictionary."""
    return _read("inventory.json")


def list_variants(*, include_deferred: bool = True) -> list[dict[str, Any]]:
    """Return variant cards, optionally excluding non-runnable entries."""
    catalog = load_catalog()
    implemented = set(catalog.get("implemented_variant_ids", []))
    cards = []
    for card in catalog.get("variants", []):
        if not include_deferred and card["variant_id"] not in implemented:
            continue
        item = deepcopy(card)
        item["runnable"] = card["variant_id"] in implemented
        cards.append(item)
    return cards


def get_variant(variant_id: str) -> dict[str, Any]:
    """Return one variant card by its exact stable identifier."""
    for card in list_variants():
        if card["variant_id"] == variant_id:
            return card
    raise UnknownVariant(f"Unknown Laboratory variant: {variant_id}")


def get_file(path: str) -> dict[str, Any]:
    """Return one archived-file record by its case-sensitive relative path."""
    for record in load_inventory().get("files", []):
        if record["path"] == path:
            item = deepcopy(record)
            item["variant"] = get_variant(record["variant_id"])
            return item
    raise CatalogError(f"Unknown archive member: {path}")


def source_to_variant() -> dict[str, str]:
    """Return the exact source-member-to-variant mapping."""
    return dict(load_catalog().get("source_to_variant", {}))


def catalog_summary() -> dict[str, Any]:
    """Return compact coverage and runnable-variant counts."""
    catalog = load_catalog()
    inventory = load_inventory()
    return {
        "archive_sha256": catalog["archive_sha256"],
        "variants": len(catalog["variants"]),
        "source_members": len(inventory["files"]),
        "parsed_members": inventory["coverage"]["python_members_parsed"],
        "implemented_variant_ids": list(catalog.get("implemented_variant_ids", [])),
        "deferred_variant_ids": list(catalog.get("deferred_variant_ids", [])),
    }
