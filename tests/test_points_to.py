# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Tests for :mod:`dynamic_call_tree_resolution.points_to`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dynamic_call_tree_resolution import (
	Address,
	FunctionPointerMember,
	FunctionSignature,
	Machine,
	Program,
	Provenance,
	Residue,
	SkipReason,
	SlotAssignment,
	assignments,
	load,
	not_enumerated,
	null_slots,
	render_path,
	unresolved_slots,
)
from tests.expected import EXPECTED_NOPIE
from tests.programs import build_program

if TYPE_CHECKING:
	from pathlib import Path


def _resolved(elf: Path) -> tuple[dict[str, SlotAssignment], Program]:
	program = load(elf)
	return {
		render_path(assignment.path): assignment for assignment in assignments(program)
	}, program


def _names(program: Program, assignment: SlotAssignment) -> set[str]:
	return {program.functions[address].name for address in assignment.candidates}


def test_nopie_resolves_every_initialized_slot(fixture_elfs: dict[str, Path]) -> None:
	resolved, program = _resolved(fixture_elfs["nopie"])
	assert set(resolved) == set(EXPECTED_NOPIE)
	for path, names in EXPECTED_NOPIE.items():
		assert _names(program, resolved[path]) == set(names)
		assert not resolved[path].relocated


def test_nopie_slots_in_writable_objects_are_ram_initializers(
	fixture_elfs: dict[str, Path],
) -> None:
	resolved, _ = _resolved(fixture_elfs["nopie"])
	assert {
		path
		for path, assignment in resolved.items()
		if assignment.provenance is Provenance.RAM_INITIALIZER
	} == {
		"holder.run",
		"node_a.fn",
		"plain_cb",
		"dev_a.ops.init",
		"dev_b.ops.init",
		"dev_c.ops.init",
		"holder2.inner.fn",
	}


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
	assert resolved["plain_cb"].relocated
	assert resolved["holder.run"].relocated
	assert not resolved["dev_a.api.open"].relocated


def test_nodebug_resolves_direct_slots_only(fixture_elfs: dict[str, Path]) -> None:
	resolved, program = _resolved(fixture_elfs["nodebug"])
	assert set(resolved) == {
		"ops_a.[0]",
		"ops_a.[1]",
		"ops_b.[0]",
		"ops_b.[1]",
		"plain_cb",
		"holder",
	}
	assert _names(program, resolved["plain_cb"]) == {"plain_target"}
	assert _names(program, resolved["holder"]) == {"undef_ptr_target"}
	assert _names(program, resolved["ops_a.[0]"]) == {"driver_a_open"}
	assert _names(program, resolved["ops_a.[1]"]) == {"driver_a_close"}
	assert _names(program, resolved["ops_b.[0]"]) == {"driver_b_open"}
	assert _names(program, resolved["ops_b.[1]"]) == {"driver_b_close"}


def test_bss_slots_are_unresolved(fixture_elfs: dict[str, Path]) -> None:
	resolved, _ = _resolved(fixture_elfs["nopie"])
	assert "bss_cb" not in resolved
	assert "bss_holder.run" not in resolved


def test_multi_tu_resolves_via_declared_types(fixture_elfs: dict[str, Path]) -> None:
	resolved, program = _resolved(fixture_elfs["multi"])
	assert set(resolved) == {
		"dev_x.api.[0]",
		"dev_x.api.[1]",
		"ops_hidden.[0]",
		"ops_hidden.[1]",
	}
	assert _names(program, resolved["dev_x.api.[0]"]) == {"hidden_open"}
	assert _names(program, resolved["dev_x.api.[1]"]) == {"hidden_close"}


def test_minimal_pie_resolves_via_relocation(fixture_elfs: dict[str, Path]) -> None:
	resolved, program = _resolved(fixture_elfs["minimal.pie"])
	assert set(resolved) >= {"ping_cb"}
	assert resolved["ping_cb"].relocated
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
	return build_program(
		Machine.EM_X86_64,
		functions=(("phantom", 0, 8), ("real_target", 0x1000, 8)),
		objects=(("fp_slot", 0x2000, stored.to_bytes(8, "little")),),
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


def test_array_globals_descend_each_element(fixture_elfs: dict[str, Path]) -> None:
	resolved, program = _resolved(fixture_elfs["arrays"])
	assert _names(program, resolved["table.[0].isr"]) == {"handler_a"}
	assert _names(program, resolved["table.[1].isr"]) == {"handler_b"}
	assert _names(program, resolved["cbs.[0]"]) == {"handler_a"}
	assert _names(program, resolved["cbs.[1]"]) == {"handler_b"}


def test_array_members_descend_each_element(fixture_elfs: dict[str, Path]) -> None:
	resolved, program = _resolved(fixture_elfs["arrays"])
	assert {
		path: _names(program, assignment)
		for path, assignment in resolved.items()
		if path.startswith("rom_bus.")
	} == {
		"rom_bus.filters.[0].rx_cb": {"handler_a"},
		"rom_bus.filters.[1].rx_cb": {"handler_b"},
		"rom_bus.hooks.[0]": {"handler_b"},
		"rom_bus.hooks.[1]": {"handler_a"},
		"rom_bus.grid.[0].[0]": {"handler_a"},
		"rom_bus.grid.[0].[1]": {"handler_b"},
		"rom_bus.grid.[1].[0]": {"handler_b"},
		"rom_bus.grid.[1].[1]": {"handler_a"},
	}


def test_anonymous_types_resolve_through_objects_and_pointers(
	fixture_elfs: dict[str, Path],
) -> None:
	resolved, program = _resolved(fixture_elfs["anonymous"])
	assert {path: _names(program, assignment) for path, assignment in resolved.items()} == {
		"anon_ops.run": {"anon_ops_fn"},
		"anon_ops_holder.ops.run": {"anon_ops_fn"},
		"bare_anon.run": {"anon_ops_fn"},
	}
	assert {
		render_path(slot.path) for slot in unresolved_slots(program, tuple(resolved.values()))
	} == {"dynamic_anon_ops.run", "visitor.visit_ops", "visitor.visit_state"}


def test_not_enumerated_lists_what_may_hold_code_without_slots(
	fixture_elfs: dict[str, Path],
) -> None:
	assert {
		(render_path(item.path), item.reason)
		for item in not_enumerated(load(fixture_elfs["arrays"]))
	} == {
		("indirection.handlers", SkipReason.POINTER_TO_POINTER),
		("indirection.filter_refs", SkipReason.POINTER_TO_POINTER),
		("indirection.filter_rows", SkipReason.POINTER_TO_ARRAY),
		("tailed_bus.tail", SkipReason.UNSIZED_ARRAY),
		("mixed_table", SkipReason.UNTYPED_OBJECT),
		("untyped_buffer", SkipReason.UNTYPED_OBJECT),
	}


def test_a_typedefd_array_member_is_a_nested_array(fixture_elfs: dict[str, Path]) -> None:
	resolved, program = _resolved(fixture_elfs["arrays"])
	assert {
		path: _names(program, assignment)
		for path, assignment in resolved.items()
		if path.startswith("indirection.")
	} == {
		"indirection.pairs.[0].[0]": {"handler_a"},
		"indirection.pairs.[0].[1]": {"handler_b"},
		"indirection.pairs.[1].[0]": {"handler_b"},
		"indirection.pairs.[1].[1]": {"handler_a"},
	}


def test_unresolved_slots_say_what_the_image_holds(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["residue"])
	resolved = tuple(
		assignment
		for assignment in assignments(program)
		if render_path(assignment.path) != "written"
	)
	assert (
		{render_path(slot.path): slot.residue for slot in unresolved_slots(program, resolved)},
		tuple(render_path(slot.path) for slot in null_slots(program, resolved)),
	) == (
		{
			"rom_arm.run": Residue.ROM_NON_FUNCTION,
			"ram_ops.stop": Residue.RAM_NULL,
			"bss_ops.run": Residue.RAM_UNINITIALIZED,
			"bss_ops.stop": Residue.RAM_UNINITIALIZED,
			"written": Residue.RAM_INITIALIZED,
		},
		("rom_ops.stop",),
	)


def test_array_globals_report_unresolved_elements(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["arrays"])
	unresolved = {
		render_path(slot.path): slot for slot in unresolved_slots(program, assignments(program))
	}
	assert set(unresolved) == {
		"dynamic_cbs.[0]",
		"dynamic_cbs.[1]",
		"ram_bus.filters.[0].rx_cb",
		"ram_bus.filters.[1].rx_cb",
		"ram_bus.hooks.[0]",
		"ram_bus.hooks.[1]",
		"ram_bus.grid.[0].[0]",
		"ram_bus.grid.[0].[1]",
		"ram_bus.grid.[1].[0]",
		"ram_bus.grid.[1].[1]",
	}
	assert unresolved["dynamic_cbs.[0]"].signature == FunctionSignature(
		return_type="void", parameters=()
	)


def test_typeless_vector_table_resolves_element_by_element(fixture_elfs: dict[str, Path]) -> None:
	resolved, program = _resolved(fixture_elfs["arrays"])
	assert _names(program, resolved["vector_table.[0]"]) == {"handler_a"}
	assert _names(program, resolved["vector_table.[1]"]) == {"handler_b"}
