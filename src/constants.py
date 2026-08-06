"""Shared reference constants for tracked items and regions.

This is the single source of truth for which items and regions the project
tracks. The collector, dashboard, drift detector, prediction scripts and the
EVE Ref importer all import from here so the labels can never disagree again.

Names are the canonical ESI ``/universe/types/{type_id}/`` names.
"""

from typing import Dict, List

# type_id -> canonical ESI name
TRACKED_ITEMS: Dict[int, str] = {
    44992: "PLEX",
    40519: "Skill Extractor",
    40520: "Large Skill Injector",
    34: "Tritanium",
    35: "Pyerite",
    36: "Mexallon",
    37: "Isogen",
    38: "Nocxium",
    39: "Zydrine",
    40: "Megacyte",
    11399: "Morphite",
}

TRACKED_TYPE_IDS: List[int] = list(TRACKED_ITEMS)

# region_id -> human readable name (trade hub in parentheses)
REGIONS: Dict[int, str] = {
    10000002: "The Forge (Jita)",
    10000043: "Domain (Amarr)",
    10000032: "Sinq Laison (Dodixie)",
}

# The Forge / Jita is the primary market hub and the default everywhere.
DEFAULT_REGION_ID = 10000002


def item_name(type_id: int) -> str:
    """Return the canonical name for a tracked type_id, or a stable fallback."""
    return TRACKED_ITEMS.get(type_id, f"Unknown Item ({type_id})")


def region_name(region_id: int) -> str:
    """Return the human readable name for a region_id, or a stable fallback."""
    return REGIONS.get(region_id, f"Region {region_id}")
