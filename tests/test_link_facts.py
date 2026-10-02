# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The qemu_cortex_m3 fixtures link with --emit-relocs and --print-gc-sections (#153)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import RelocationSection

from dynamic_call_tree_resolution import load
from dynamic_call_tree_resolution.vsa import address_taken

if TYPE_CHECKING:
	from pathlib import Path

pytestmark = pytest.mark.image


@pytest.mark.parametrize(
	("name", "removed"), [("hello", 632), ("sensor-two-impl", 615), ("sensor-threads", 616)]
)
def test_each_cortex_m3_fixture_ships_its_map_and_its_final_links_gc_listing(
	zephyr_fixtures: Path, name: str, removed: int
) -> None:
	zephyr = zephyr_fixtures / name / "zephyr"
	listing = (zephyr / "gc-sections.txt").read_text().splitlines()
	assert (
		(zephyr / "zephyr_final.map").is_file(),
		len(listing),
		[line for line in listing if not line.startswith("removing unused section '")],
	) == (True, removed, [])


def test_sensor_two_impl_keeps_its_static_relocations_and_loads_the_same_address_taken_set(
	zephyr_fixtures: Path,
) -> None:
	elf = zephyr_fixtures / "sensor-two-impl" / "zephyr" / "zephyr.elf"
	with elf.open("rb") as stream:
		kept = sum(
			isinstance(section, RelocationSection) for section in ELFFile(stream).iter_sections()
		)
	assert (kept, len(address_taken(load(elf)))) == (17, 55)
