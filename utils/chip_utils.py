#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Tiny helpers so we never hand-craft chip-aware names again."""

def chip_prefix(chip_id: int) -> str:
    return f"Chip{chip_id}/"

def chipify(base: str, chip_id: int = 0) -> str:
    """Attach `<ChipN>/` in front of a pre-existing accelerator name."""
    return chip_prefix(chip_id) + base

def dechip(name: str) -> str:
    """Strip the `<ChipN>/` prefix (if any).  Used whenever we test .startswith()."""
    return name.split('/', 1)[-1]

def chip_id_from_name(name: str) -> int:
    """Return the numeric chip id; 0 if none."""
    if '/' not in name:
        return 0
    return int(name.split('/', 1)[0][4:])   # "Chip<N>"
