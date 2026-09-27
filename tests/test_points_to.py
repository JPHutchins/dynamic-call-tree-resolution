# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Tests for :mod:`dynamic_call_tree_resolution.points_to`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dynamic_call_tree_resolution import (
	Address,
	DataObject,
	Function,
	FunctionPointerMember,
	FunctionSignature,
	Program,
	Provenance,
	SlotAssignment,
	assignments,
	load,
	render_path,
	unresolved_slots,
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
	"dev_c.api.open": ("driver_b_open",),
	"dev_c.api.close": ("driver_b_close",),
	"dev_c.context.open": ("driver_a_open",),
	"dev_c.context.close": ("driver_a_close",),
	"dev_a.ops.init": ("dev_init",),
	"dev_b.ops.init": ("dev_init",),
	"dev_c.ops.init": ("dev_init",),
	"holder2.inner.fn": ("anon_fn",),
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


def test_multi_tu_resolves_via_declared_types(fixture_elfs: dict[str, Path]) -> None:
	resolved, program = _resolved(fixture_elfs["multi"])
	assert set(resolved) == {"dev_x.api", "ops_hidden"}
	assert _names(program, resolved["dev_x.api"]) == {"hidden_open"}


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


def test_unresolved_slots_report_bss_slots_with_signatures(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["nopie"])
	by_path = {
		render_path(slot.path): slot for slot in unresolved_slots(program, assignments(program))
	}
	assert set(by_path) == {"bss_cb", "bss_holder.run"}
	assert by_path["bss_cb"].signature == FunctionSignature(return_type="void", parameters=("int",))
	assert by_path["bss_holder.run"].signature == FunctionSignature(
		return_type="void", parameters=()
	)


@pytest.mark.parametrize("variant", ["pie", "o2"])
def test_unresolved_slots_across_variants(fixture_elfs: dict[str, Path], variant: str) -> None:
	program = load(fixture_elfs[variant])
	assert {render_path(slot.path) for slot in unresolved_slots(program, assignments(program))} == {
		"bss_cb",
		"bss_holder.run",
	}


def test_unresolved_slots_without_dwarf_reports_nothing(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["nodebug"])
	assert unresolved_slots(program, assignments(program)) == ()


def test_null_fixture_keeps_no_functions_at_address_zero(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["null"])
	assert Address(0) not in program.functions
	assert Address(1) not in program.functions


def test_null_fixture_resolves_nothing(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["null"])
	assert assignments(program) == ()
	assert {render_path(slot.path) for slot in unresolved_slots(program, assignments(program))} == {
		"null_cb"
	}


def _program_with_slot(stored: int) -> Program:
	return Program(
		byte_order="little",
		pointer_size=8,
		machine="EM_X86_64",
		functions={
			Address(0): Function(name="phantom", address=Address(0), size=8, signature=None),
			Address(0x1000): Function(
				name="real_target", address=Address(0x1000), size=8, signature=None
			),
		},
		objects={
			Address(0x2000): DataObject(
				name="fp_slot", address=Address(0x2000), size=8, type_name=None, signature=None
			)
		},
		layouts={},
		relocations=(),
		sections={Address(0x2000): stored.to_bytes(8, "little")},
	)


def test_null_slot_does_not_resolve_to_a_function_at_address_zero() -> None:
	assert assignments(_program_with_slot(0)) == ()


def test_declaration_only_struct_dies_do_not_clobber_definitions(
	fixture_elfs: dict[str, Path],
) -> None:
	program = load(fixture_elfs["decl"])
	assert program.layouts["struct shared"].members == (
		FunctionPointerMember(
			kind="function_pointer",
			name="fn",
			offset=0,
			signature=FunctionSignature(return_type="int", parameters=()),
		),
		FunctionPointerMember(
			kind="function_pointer",
			name="extra",
			offset=8,
			signature=FunctionSignature(return_type="int", parameters=()),
		),
	)
	resolved, program = _resolved(fixture_elfs["decl"])
	assert _names(program, resolved["instance.fn"]) == {"target_fn"}
	assert _names(program, resolved["second_instance.fn"]) == {"second_fn"}
	assert _names(program, resolved["second_instance.extra"]) == {"second_fn"}


def test_nonzero_slot_still_resolves_even_with_a_function_at_zero() -> None:
	resolved = assignments(_program_with_slot(0x1000))
	assert [assignment.candidates for assignment in resolved] == [frozenset({Address(0x1000)})]
