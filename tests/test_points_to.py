# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Tests for :mod:`dynamic_call_tree_resolution.points_to`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dynamic_call_tree_resolution import (
	Program,
	Provenance,
	SlotAssignment,
	assignments,
	load,
	render_path,
)

if TYPE_CHECKING:
	from pathlib import Path

EXPECTED_NOPIE: dict[str, tuple[str, ...]] = {
	"ops_a.open": ("driver_a_open",),
	"ops_a.close": ("driver_a_close",),
	"ops_b.open": ("driver_b_open",),
	"ops_b.close": ("driver_b_close",),
	"dev_a.api.open": ("driver_a_open",),
	"dev_a.api.close": ("driver_a_close",),
	"dev_b.api.open": ("driver_b_open",),
	"dev_b.api.close": ("driver_b_close",),
	"holder.run": ("undef_ptr_target",),
	"node_a.fn": ("node_fn",),
	"plain_cb": ("plain_target",),
}


def _resolved(elf: Path) -> tuple[dict[str, SlotAssignment], Program]:
	program = load(elf)
	return {
		render_path(assignment.path): assignment for assignment in assignments(program)
	}, program


def _names(program: Program, assignment: SlotAssignment) -> set[str]:
	return {program.functions[address].name for address in assignment.candidates}


def test_nopie_resolves_all_const_slots_exactly(fixture_elfs: dict[str, Path]) -> None:
	resolved, program = _resolved(fixture_elfs["nopie"])
	assert set(resolved) == set(EXPECTED_NOPIE)
	for path, names in EXPECTED_NOPIE.items():
		assert _names(program, resolved[path]) == set(names)
		assert resolved[path].provenance is Provenance.CONSTANT_DATA


@pytest.mark.parametrize("variant", ["o2", "dwarf4"])
def test_other_build_modes_match_nopie(
	fixture_elfs: dict[str, Path],
	variant: str,
) -> None:
	resolved, program = _resolved(fixture_elfs[variant])
	assert set(resolved) == set(EXPECTED_NOPIE)
	for path, names in EXPECTED_NOPIE.items():
		assert _names(program, resolved[path]) == set(names)


def test_pie_resolves_relocations_with_their_slots(fixture_elfs: dict[str, Path]) -> None:
	resolved, _ = _resolved(fixture_elfs["pie"])
	assert set(resolved) >= set(EXPECTED_NOPIE)
	assert resolved["plain_cb"].provenance is Provenance.RELOCATION
	assert resolved["holder.run"].provenance is Provenance.RELOCATION
	assert resolved["dev_a.api.open"].provenance is Provenance.CONSTANT_DATA


def test_nodebug_resolves_direct_slots_only(fixture_elfs: dict[str, Path]) -> None:
	resolved, program = _resolved(fixture_elfs["nodebug"])
	assert set(resolved) == {"ops_a", "ops_b", "plain_cb", "holder"}
	assert _names(program, resolved["plain_cb"]) == {"plain_target"}
	assert _names(program, resolved["holder"]) == {"undef_ptr_target"}
	assert _names(program, resolved["ops_a"]) == {"driver_a_open"}


def test_bss_slots_are_unresolved(fixture_elfs: dict[str, Path]) -> None:
	resolved, _ = _resolved(fixture_elfs["nopie"])
	assert "bss_cb" not in resolved
	assert "bss_holder.run" not in resolved


def test_minimal_pie_resolves_via_relocation(fixture_elfs: dict[str, Path]) -> None:
	resolved, program = _resolved(fixture_elfs["minimal.pie"])
	assert set(resolved) >= {"ping_cb"}
	assert resolved["ping_cb"].provenance is Provenance.RELOCATION
	assert _names(program, resolved["ping_cb"]) == {"ping"}
	assert "undef_ptr" not in resolved


def test_assignments_are_sorted_by_slot(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["nopie"])
	resolved = assignments(program)
	assert [assignment.slot for assignment in resolved] == sorted(
		assignment.slot for assignment in resolved
	)
