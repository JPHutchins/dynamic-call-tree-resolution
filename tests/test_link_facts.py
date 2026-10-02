# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The qemu_cortex_m3 fixtures link with --emit-relocs and --print-gc-sections (#153)."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import pytest
from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import RelocationSection
from elftools.elf.sections import SymbolTableSection
from salix import replace

from dynamic_call_tree_resolution import ReferenceKind, load
from dynamic_call_tree_resolution.points_to import read_pointer
from dynamic_call_tree_resolution.report import referrers_report
from dynamic_call_tree_resolution.rtos import RtosChoice, rtos_model
from dynamic_call_tree_resolution.vsa import address_taken, linked_address_taken
from dynamic_call_tree_resolution.vsa.fallback import referenced_only_at, referrers

if TYPE_CHECKING:
	from pathlib import Path

	from dynamic_call_tree_resolution.model import Program

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


@pytest.mark.parametrize(
	("name", "taken"), [("hello", 26), ("sensor-two-impl", 55), ("sensor-threads", 57)]
)
def test_the_linked_address_taken_set_equals_the_byte_scans_on_each_cortex_m3_fixture(
	zephyr_fixtures: Path, name: str, taken: int
) -> None:
	program = load(zephyr_fixtures / name / "zephyr" / "zephyr.elf")
	assert (
		len(linked_address_taken(program)),
		linked_address_taken(program) == address_taken(replace(program, link_references=())),
		address_taken(program) == linked_address_taken(program),
	) == (taken, True, True)


def test_sensor_two_impl_keeps_references_of_each_kind_and_none_of_its_section_references_is_a_function(
	zephyr_fixtures: Path,
) -> None:
	program = load(zephyr_fixtures / "sensor-two-impl" / "zephyr" / "zephyr.elf")
	section_addresses = [
		reference
		for reference in program.link_references
		if reference.kind is ReferenceKind.ADDRESS and not reference.symbol
	]
	assert (
		Counter(reference.kind for reference in program.link_references),
		len(section_addresses),
		[
			reference
			for reference in section_addresses
			if read_pointer(program, reference.slot, None) in program.functions
		],
	) == (
		Counter({ReferenceKind.ADDRESS: 414, ReferenceKind.CALL: 330, ReferenceKind.OTHER: 2}),
		174,
		[],
	)


def test_the_vector_tables_reference_to_address_zero_is_not_the_absolute_vfscanf_at_one(
	zephyr_fixtures: Path,
) -> None:
	elf = zephyr_fixtures / "sensor-two-impl" / "zephyr" / "zephyr.elf"
	with elf.open("rb") as stream:
		symbols = ELFFile(stream).get_section_by_name(".symtab")
		assert isinstance(symbols, SymbolTableSection)
		vfscanf = [
			(symbol["st_value"], symbol["st_shndx"])
			for symbol in symbols.iter_symbols()
			if symbol.name == "vfscanf"
		]
	program = load(elf)
	vector_start = replace(
		program,
		link_references=tuple(
			reference
			for reference in program.link_references
			if reference.symbol == "_vector_start"
		),
	)
	assert (
		vfscanf,
		[(reference.value, reference.kind) for reference in vector_start.link_references],
		linked_address_taken(vector_start),
	) == ([(1, "SHN_ABS")], [(0, ReferenceKind.ADDRESS)], frozenset())


def test_a_link_without_emit_relocs_has_no_link_references(zephyr_fixtures: Path) -> None:
	program = load(zephyr_fixtures / "counter-su" / "zephyr" / "zephyr.exe")
	assert (program.link_references, len(address_taken(program))) == ((), 236)


def _seeded_entries(program: Program) -> list[str]:
	threads = rtos_model(program, RtosChoice.AUTO).threads
	slots = {
		thread.entry: frozenset(
			other.entry_slot for other in threads if other.entry == thread.entry
		)
		for thread in threads
	}
	return sorted(
		program.functions[entry].name
		for entry in referenced_only_at(referrers(program, slots.keys()), slots)
	)


@pytest.mark.parametrize(
	("name", "seeded"),
	[
		("sensor-two-impl", ["motion_thread", "thermal_thread"]),
		("sensor-threads", ["sensor_thread"]),
	],
)
def test_the_references_the_linker_kept_make_each_fixtures_seed_decisions_the_byte_scan_makes(
	zephyr_fixtures: Path, name: str, seeded: list[str]
) -> None:
	program = load(zephyr_fixtures / name / "zephyr" / "zephyr.elf")
	assert (
		_seeded_entries(program),
		_seeded_entries(replace(program, link_references=())),
	) == (seeded, seeded)


def test_sensor_threads_names_who_holds_each_thread_entry(zephyr_fixtures: Path) -> None:
	program = load(zephyr_fixtures / "sensor-threads" / "zephyr" / "zephyr.elf")
	report = referrers_report(program)
	assert (
		len(report),
		all(row.referrers for row in report),
		{
			row.name: tuple(referrer.holder for referrer in row.referrers)
			for row in report
			if row.name in {"bg_thread_main", "idle", "sensor_thread", "z_thread_entry"}
		},
	) == (
		len(address_taken(program)),
		True,
		{
			"bg_thread_main": ("z_cstart",),
			"idle": ("z_init_cpu",),
			"sensor_thread": ("_k_thread_data_motion_tid", "_k_thread_data_thermal_tid"),
			"z_thread_entry": ("arch_new_thread", "arch_switch_to_main_thread"),
		},
	)
