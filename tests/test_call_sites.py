# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Tests for :mod:`dynamic_call_tree_resolution.call_sites`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dynamic_call_tree_resolution import (
	Address,
	DataObject,
	Function,
	FunctionSignature,
	Program,
	Provenance,
	SlotAssignment,
	assignments,
	build_report,
	call_site_candidates,
	extract_call_sites,
	load,
	matching_targets,
	per_caller_candidates,
)

if TYPE_CHECKING:
	from pathlib import Path


def _program(
	machine: str,
	code: bytes,
	*,
	functions: tuple[tuple[str, int, int], ...] = (),
	objects: tuple[tuple[str, int, bytes], ...] = (),
	pointer_size: int = 8,
) -> Program:
	return Program(
		byte_order="little",
		pointer_size=pointer_size,
		machine=machine,
		functions={
			Address(address): Function(
				name=name, address=Address(address), size=size, signature=None
			)
			for name, address, size in functions
		},
		objects={
			Address(address): DataObject(
				name=name,
				address=Address(address),
				size=len(data),
				type_name=None,
				signature=None,
			)
			for name, address, data in objects
		},
		layouts={},
		relocations=(),
		sections=(
			{Address(0x1000): code} | {Address(address): data for _, address, data in objects}
		),
	)


def _pointer(value: int, size: int) -> bytes:
	return value.to_bytes(size, "little")


def _x86(code: bytes, *, objects: tuple[tuple[str, int, bytes], ...] = ()) -> Program:
	return _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
		objects=objects,
	)


def test_x86_memory_operand_site_resolves_through_the_slot() -> None:
	program = _x86(
		bytes.fromhex("ff 15 fa 0f 00 00"), objects=(("slot", 0x2000, _pointer(0x3000, 8)),)
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1000
	assert site.slot == 0x2000
	assert call_site_candidates(program, site, {}) == frozenset({Address(0x3000)})


def test_x86_register_load_chain_dereferences() -> None:
	program = _x86(
		bytes.fromhex("48 8b 05 f9 0f 00 00ff d0"), objects=(("slot", 0x2000, _pointer(0x3000, 8)),)
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1007
	assert site.slot == 0x3000
	assert call_site_candidates(program, site, {}) == frozenset({Address(0x3000)})


def test_x86_untracked_register_is_unresolved() -> None:
	program = _x86(bytes.fromhex("ff d0"))
	(site,) = extract_call_sites(program)
	assert site.slot is None
	assert call_site_candidates(program, site, {}) == frozenset()


def test_x86_direct_call_in_window_clears_registers() -> None:
	program = _x86(
		bytes.fromhex("48 8b 05 f9 0f 00 00e8 00 00 00 00ff d0"),
		objects=(("slot", 0x2000, _pointer(0x3000, 8)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100C
	assert site.slot is None


def test_x86_callee_saved_register_survives_a_direct_call() -> None:
	code = bytes.fromhex("48 c7 c3 00 30 00 00e8 00 00 00 00ff d3")
	program = _x86(code)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100C
	assert site.slot == 0x3000


def test_x86_32_direct_call_in_window_clears_caller_saved_registers() -> None:
	code = bytes.fromhex("b8 00 30 00 00e8 00 00 00 00ff d0")
	program = _program(
		"EM_386",
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100A
	assert site.slot is None


def test_x86_32_callee_saved_register_survives_a_direct_call() -> None:
	code = bytes.fromhex("bb 00 30 00 00e8 00 00 00 00ff d3")
	program = _program(
		"EM_386",
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100A
	assert site.slot == 0x3000


def test_x86_jump_to_the_site_carries_state() -> None:
	code = bytes.fromhex("48 c7 c3 00 30 00 00eb 00ff d3")
	program = _x86(code)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1009
	assert site.slot == 0x3000


def test_x86_indirect_jump_in_window_kills_the_stream() -> None:
	code = bytes.fromhex("48 b8 00 30 00 00 00 00 00 00ff e0ff d0")
	program = _x86(code)
	jmp_site, call_site = extract_call_sites(program)
	assert jmp_site.site_address == 0x100A
	assert jmp_site.slot == 0x3000
	assert call_site.site_address == 0x100C
	assert call_site.slot is None


def test_x86_branch_before_the_window_restarts_it() -> None:
	code = b"\x90" * 11 + bytes.fromhex("48 c7 c3 00 30 00 0048 85 c074 e9ff d3")
	program = _x86(code)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1017
	assert site.slot == 0x3000


def test_x86_unresolvable_lea_clears_the_destination() -> None:
	code = bytes.fromhex("48 8d 45 f8ff d0")
	program = _x86(code)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1004
	assert site.slot is None


def test_x86_loop_back_edge_carries_state() -> None:
	code = bytes.fromhex("48 c7 c3 00 30 00 0048 85 c075 0548 ff c8eb f6ff d3")
	program = _x86(code)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1011
	assert site.slot == 0x3000


def test_x86_conditional_paths_union_into_the_site_candidates() -> None:
	code = bytes.fromhex("48 c7 c3 00 30 00 0048 85 c074 0748 c7 c3 00 40 00 00ff d3")
	program = _program(
		"EM_X86_64",
		code,
		functions=(
			("caller", 0x1000, len(code)),
			("target_a", 0x3000, 1),
			("target_b", 0x4000, 1),
		),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1013
	assert site.slot is None
	assert site.candidates == frozenset({Address(0x3000), Address(0x4000)})


def test_conflicting_paths_fall_back_to_the_resolved_union() -> None:
	code = bytes.fromhex("48 c7 c3 00 30 00 0048 85 c074 0748 c7 c3 00 40 00 00ff d3")
	program = _program(
		"EM_X86_64",
		code,
		functions=(
			("caller", 0x1000, len(code)),
			("target_a", 0x3000, 1),
			("target_b", 0x4000, 1),
		),
	)
	resolved = (
		SlotAssignment(
			slot=Address(0x3000),
			path=("target_a",),
			candidates=frozenset({Address(0x3000)}),
			provenance=Provenance.CONSTANT_DATA,
		),
		SlotAssignment(
			slot=Address(0x4000),
			path=("target_b",),
			candidates=frozenset({Address(0x4000)}),
			provenance=Provenance.CONSTANT_DATA,
		),
	)
	by_caller, fallback = per_caller_candidates(program, extract_call_sites(program), resolved)
	assert by_caller == {"caller": frozenset({"target_a", "target_b"})}
	assert fallback == frozenset({"target_a", "target_b"})


def test_x86_conditional_branch_keeps_registers() -> None:
	program = _x86(
		bytes.fromhex("48 8b 05 f9 0f 00 0048 85 c074 02ff d0"),
		objects=(("slot", 0x2000, _pointer(0x3000, 8)),),
	)
	(site,) = extract_call_sites(program)
	assert site.slot == 0x3000


def test_x86_jmp_tail_site() -> None:
	program = _x86(
		bytes.fromhex("48 8b 05 f9 0f 00 00ff e0"), objects=(("slot", 0x2000, _pointer(0x3000, 8)),)
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1007
	assert site.slot == 0x3000


def test_x86_absolute_memory_operand() -> None:
	program = _x86(
		bytes.fromhex("ff 14 25 00 20 00 00"), objects=(("slot", 0x2000, _pointer(0x3000, 8)),)
	)
	(site,) = extract_call_sites(program)
	assert site.slot == 0x2000


def test_x86_indexed_memory_operand_is_unresolved() -> None:
	program = _x86(bytes.fromhex("ff 14 48"))
	(site,) = extract_call_sites(program)
	assert site.slot is None


def test_x86_frame_relative_operand_is_unresolved() -> None:
	program = _x86(bytes.fromhex("ff 55 f8"))
	(site,) = extract_call_sites(program)
	assert site.slot is None


def test_x86_register_to_register_move() -> None:
	program = _x86(
		bytes.fromhex("48 8b 05 f9 0f 00 0048 89 c2ff d2"),
		objects=(("slot", 0x2000, _pointer(0x3000, 8)),),
	)
	(site,) = extract_call_sites(program)
	assert site.slot == 0x3000


def test_x86_lea_then_memory_call() -> None:
	program = _x86(
		bytes.fromhex("48 8d 05 f9 0f 00 00ff 10"), objects=(("slot", 0x2000, _pointer(0x3000, 8)),)
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1007
	assert site.slot == 0x2000
	assert call_site_candidates(program, site, {}) == frozenset({Address(0x3000)})


def test_x86_immediate_load_resolves_directly() -> None:
	program = _x86(bytes.fromhex("48 b8 00 30 00 00 00 00 00 00ff d0"))
	(site,) = extract_call_sites(program)
	assert site.slot == 0x3000
	assert call_site_candidates(program, site, {}) == frozenset({Address(0x3000)})


def test_x86_elementwise_add_shifts_the_tracked_address() -> None:
	program = _x86(
		bytes.fromhex("48 8b 05 f9 0f 00 0048 83 c0 04ff d0"),
		objects=(("slot", 0x2000, _pointer(0x3000, 8)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100B
	assert site.slot == 0x3004
	assert call_site_candidates(program, site, {}) == frozenset()


def test_x86_elementwise_add_lands_on_a_resolvable_slot() -> None:
	program = _x86(
		bytes.fromhex("48 c7 c0 00 20 00 0048 83 c0 04ff 10"),
		objects=(("slot", 0x2004, _pointer(0x3000, 8)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100B
	assert site.slot == 0x2004
	assert call_site_candidates(program, site, {}) == frozenset({Address(0x3000)})


def test_x86_definitions_beyond_the_former_window_resolve() -> None:
	program = _x86(
		bytes.fromhex("48 b8 00 30 00 00 00 00 00 00") + b"\x90" * 12 + bytes.fromhex("ff d0")
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1016
	assert site.slot == 0x3000
	assert site.candidates == frozenset({Address(0x3000)})


def test_arm_thumb_bit_twins_are_deduplicated() -> None:
	body = bytes.fromhex("00 4b 98 47") + _pointer(0x2000, 4)
	program = _program(
		"EM_ARM",
		body,
		functions=(
			("caller", 0x1000, len(body)),
			("caller", 0x1001, len(body)),
			("target", 0x2000, 4),
		),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x1000
	assert site.site_address == 0x1002
	assert site.slot == 0x2000


def test_arm_thumb_bit_only_symbol_decodes_aligned() -> None:
	body = bytes.fromhex("00 4b 98 47") + _pointer(0x2000, 4)
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1001, len(body)), ("target", 0x2000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x1001
	assert site.site_address == 0x1002
	assert site.slot == 0x2000


def test_arm_literal_pool_load_resolves() -> None:
	body = bytes.fromhex("02 4b 98 47") + b"\x00\xbf" * 4 + _pointer(0x2000, 4)
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x2000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1002
	assert site.slot == 0x2000
	assert call_site_candidates(program, site, {}) == frozenset({Address(0x2000)})


def test_arm_register_load_chain_dereferences_data() -> None:
	body = (
		bytes.fromhex("01 4b 18 68 80 4700 bf")
		+ _pointer(0x3000, 4)  # ldr r3,[pc,#4]; ldr r0,[r3]; blx r0; pool=0x3000
	)
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x2000, 4)),
		objects=(("slot", 0x3000, _pointer(0x2000, 4)),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1004
	assert site.slot == 0x2000
	assert call_site_candidates(program, site, {}) == frozenset({Address(0x2000)})


def test_site_slot_joins_resolved_assignments() -> None:
	body = bytes.fromhex("00 4b 98 47") + _pointer(0x3000, 4)
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x2000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.slot == 0x3000
	assignment = SlotAssignment(
		slot=Address(0x3000),
		path=("holder", "run"),
		candidates=frozenset({Address(0x2000)}),
		provenance=Provenance.CONSTANT_DATA,
	)
	assert call_site_candidates(program, site, {Address(0x3000): assignment}) == frozenset(
		{Address(0x2000)}
	)


def test_arm_bx_lr_is_a_return_not_a_site() -> None:
	program = _program(
		"EM_ARM", bytes.fromhex("70 47"), functions=(("caller", 0x1000, 2),), pointer_size=4
	)
	assert extract_call_sites(program) == ()


def test_arm_bx_tail_site() -> None:
	body = bytes.fromhex("00 4b 18 47") + _pointer(0x2000, 4)
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x2000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1002
	assert site.slot == 0x2000


def test_arm_register_moves() -> None:
	body = bytes.fromhex("01 4b 18 46 01 21 80 47") + _pointer(0x2000, 4)
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x2000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1006
	assert site.slot == 0x2000


def test_arm_stack_load_is_unresolved() -> None:
	program = _program(
		"EM_ARM", bytes.fromhex("00 98 80 47"), functions=(("caller", 0x1000, 4),), pointer_size=4
	)
	(site,) = extract_call_sites(program)
	assert site.slot is None


def test_arm_load_with_untracked_base_is_unresolved() -> None:
	program = _program(
		"EM_ARM", bytes.fromhex("18 68 80 47"), functions=(("caller", 0x1000, 4),), pointer_size=4
	)
	(site,) = extract_call_sites(program)
	assert site.slot is None


def test_arm_indexed_load_is_unresolved() -> None:
	program = _program(
		"EM_ARM", bytes.fromhex("88 58 80 47"), functions=(("caller", 0x1000, 4),), pointer_size=4
	)
	(site,) = extract_call_sites(program)
	assert site.slot is None


def test_arm_callee_saved_register_survives_a_bl() -> None:
	body = bytes.fromhex("02 4c00 f0 00 f8a0 47") + b"\x00\xbf" * 2 + _pointer(0x3000, 4)
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x2000, 4)),
		objects=(("slot", 0x3000, _pointer(0x2000, 4)),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1006
	assert site.slot == 0x3000
	assert call_site_candidates(program, site, {}) == frozenset({Address(0x2000)})


def test_arm_direct_branch_to_the_site_carries_state() -> None:
	body = bytes.fromhex("02 4c00 e0a0 47") + b"\x00\xbf" * 3 + _pointer(0x3000, 4)
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x2000, 4)),
		objects=(("slot", 0x3000, _pointer(0x2000, 4)),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1004
	assert site.slot == 0x3000
	assert call_site_candidates(program, site, {}) == frozenset({Address(0x2000)})


def test_arm_conditional_branch_past_the_site_keeps_fallthrough_state() -> None:
	body = bytes.fromhex("02 4ca4 4201 d0a0 47") + b"\x00\xbf" * 2 + _pointer(0x3000, 4)
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x2000, 4)),
		objects=(("slot", 0x3000, _pointer(0x2000, 4)),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1006
	assert site.slot == 0x3000
	assert call_site_candidates(program, site, {}) == frozenset({Address(0x2000)})


def test_arm_movw_movt_pair_resolves() -> None:
	body = bytes.fromhex("41 f2 34 23c5 f2 78 6398 47")
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x56781234, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1008
	assert site.slot == 0x56781234
	assert call_site_candidates(program, site, {}) == frozenset({Address(0x56781234)})


def test_arm_movt_without_movw_is_unresolved() -> None:
	body = bytes.fromhex("c5 f2 78 6398 47")
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1004
	assert site.slot is None
	assert site.candidates == frozenset()


def test_arm_cbz_past_the_site_keeps_fallthrough_state() -> None:
	body = bytes.fromhex("02 4c0c b3a0 47") + b"\x00\xbf" * 3 + _pointer(0x3000, 4)
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x2000, 4)),
		objects=(("slot", 0x3000, _pointer(0x2000, 4)),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1004
	assert site.slot == 0x3000
	assert call_site_candidates(program, site, {}) == frozenset({Address(0x2000)})


def test_arm_call_in_window_clears_registers() -> None:
	body = bytes.fromhex("00 4b 00 f0 00 f8 98 47") + _pointer(0x2000, 4)
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x2000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1006
	assert site.slot is None


def test_arm_elementwise_add_shifts_the_tracked_address() -> None:
	body = bytes.fromhex("01 4b 04 33 98 4700 bf") + _pointer(0x2000, 4)
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x2000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1004
	assert site.slot == 0x2004
	assert call_site_candidates(program, site, {}) == frozenset()


def test_chasing_self_referential_pointer_stops() -> None:
	program = _x86(
		bytes.fromhex("ff 15 fa 0f 00 00"), objects=(("self", 0x2000, _pointer(0x2000, 8)),)
	)
	(site,) = extract_call_sites(program)
	assert call_site_candidates(program, site, {}) == frozenset()


def test_skipped_bytes_in_the_window_are_ignored() -> None:
	body = bytes.fromhex("df f7 02 4b 98 47") + b"\x00\xbf" * 3 + _pointer(0x2000, 4)
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x2000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1004
	assert site.slot == 0x2000


def test_skipped_bytes_are_not_sites() -> None:
	body = bytes.fromhex("02 4b 98 47 df f7") + b"\x00\xbf" * 3 + _pointer(0x2000, 4)
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x2000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1002
	assert site.slot == 0x2000


def test_unsupported_machine_has_no_sites() -> None:
	assert extract_call_sites(_program("EM_RISCV", b"")) == ()


def test_function_without_code_bytes_has_no_sites() -> None:
	program = _program("EM_X86_64", b"", functions=(("bare", 0x5000, 4),))
	assert extract_call_sites(program) == ()


def test_fixture_main_sites_resolve_nopie(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["nopie"])
	resolved = {assignment.slot: assignment for assignment in assignments(program)}
	main = next(function for function in program.functions.values() if function.name == "main")
	names = [
		{
			program.functions[address].name
			for address in call_site_candidates(program, site, resolved)
		}
		for site in extract_call_sites(program)
		if site.caller_address == main.address
	]
	assert names == [
		{"plain_target"},
		set(),
		{"driver_a_open"},
		{"driver_b_close"},
		{"undef_ptr_target"},
		{"node_fn"},
	]


@pytest.mark.parametrize("variant", ["pie", "o2", "nodebug"])
def test_fixture_main_sites_resolve_across_variants(
	fixture_elfs: dict[str, Path], variant: str
) -> None:
	program = load(fixture_elfs[variant])
	resolved = {assignment.slot: assignment for assignment in assignments(program)}
	main = next(function for function in program.functions.values() if function.name == "main")
	names = [
		{
			program.functions[address].name
			for address in call_site_candidates(program, site, resolved)
		}
		for site in extract_call_sites(program)
		if site.caller_address == main.address
	]
	assert len(names) == 6
	assert sum(not candidate_names for candidate_names in names) == 1  # bss_cb
	assert {
		"plain_target",
		"driver_a_open",
		"driver_b_close",
		"undef_ptr_target",
		"node_fn",
	} <= set().union(*names)


def test_per_caller_candidates_unions_sites_with_fallback(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["nopie"])
	resolved = assignments(program)
	by_caller, fallback = per_caller_candidates(program, extract_call_sites(program), resolved)
	assert set(by_caller) <= {function.name for function in program.functions.values()}
	assert by_caller["main"] == fallback  # main's bss_cb site pulls in the fallback union
	assert fallback == {
		program.functions[address].name
		for assignment in resolved
		for address in assignment.candidates
	}


def test_x86_loop_merges_conditional_paths_into_a_union() -> None:
	code = bytes.fromhex("48 c7 c3 00 30 00 0048 85 c074 0a48 c7 c3 00 40 00 0048 ff c975 efff d3")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)), ("first", 0x3000, 1), ("second", 0x4000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1018
	assert site.candidates == frozenset({Address(0x3000), Address(0x4000)})
	assert call_site_candidates(program, site, {}) == frozenset({Address(0x3000), Address(0x4000)})


def test_x86_loop_carried_union_overflows_k_to_top() -> None:
	table = b"".join((0x3000 + index * 8).to_bytes(8, "little") for index in range(65))
	code = bytes.fromhex(
		"48 c7 c1 41 00 00 0048 c7 c3 00 20 00 0048 8b 0348 83 c3 0848 ff c975 f4ff d0"
	)
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)),),
		objects=(("table", 0x2000, table),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x101A
	assert site.candidates == frozenset()


def test_x86_top_index_over_baked_object_enumerates_its_slots() -> None:
	code = bytes.fromhex("48 c7 c3 00 20 00 0048 8b 04 cbff d0")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)), ("first", 0x3000, 1), ("second", 0x4000, 1)),
		objects=(("table", 0x2000, _pointer(0x3000, 8) + _pointer(0x4000, 8)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100B
	assert site.candidates == frozenset({Address(0x3000), Address(0x4000)})


def test_x86_top_index_without_an_enclosing_object_is_unresolved() -> None:
	code = bytes.fromhex("48 c7 c3 00 30 00 0048 8b 04 cbff d0")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100B
	assert site.candidates == frozenset()


def test_x86_top_index_over_a_too_small_object_is_unresolved() -> None:
	program = Program(
		byte_order="little",
		pointer_size=8,
		machine="EM_X86_64",
		functions={
			Address(0x1000): Function(
				name="caller", address=Address(0x1000), size=15, signature=None
			),
		},
		objects={
			Address(0x2000): DataObject(
				name="short", address=Address(0x2000), size=2, type_name=None, signature=None
			)
		},
		layouts={},
		relocations=(),
		sections={Address(0x1000): bytes.fromhex("48 c7 c3 00 20 00 0048 8b 04 cbff d0")},
	)
	(site,) = extract_call_sites(program)
	assert site.candidates == frozenset()


def test_x86_bss_slot_read_is_unknown() -> None:
	program = Program(
		byte_order="little",
		pointer_size=8,
		machine="EM_X86_64",
		functions={
			Address(0x1000): Function(
				name="caller", address=Address(0x1000), size=9, signature=None
			),
		},
		objects={
			Address(0x2000): DataObject(
				name="bss_slot", address=Address(0x2000), size=8, type_name=None, signature=None
			)
		},
		layouts={},
		relocations=(),
		sections={Address(0x1000): bytes.fromhex("48 8b 05 f9 0f 00 00ff d0")},
	)
	(site,) = extract_call_sites(program)
	assert site.candidates == frozenset()


def test_x86_read_of_unmapped_memory_drops_the_value() -> None:
	code = bytes.fromhex("48 8b 05 f9 07 00 00ff d0")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)),),
	)
	(site,) = extract_call_sites(program)
	assert site.candidates == frozenset()


def test_x86_join_with_one_path_missing_the_register_is_top() -> None:
	code = bytes.fromhex("48 c7 c0 00 30 00 0048 85 c074 0748 c7 c3 00 40 00 00eb 02ff d3")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1015
	assert site.candidates == frozenset()


def test_x86_join_with_the_other_path_missing_the_register_is_top() -> None:
	code = bytes.fromhex("48 c7 c0 00 30 00 0048 85 c075 0748 c7 c3 00 40 00 00eb 02ff d3")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1015
	assert site.candidates == frozenset()


def test_x86_loop_instruction_top_out_the_counter_and_keep_values() -> None:
	code = bytes.fromhex("b9 03 00 00 0048 c7 c0 00 30 00 00e2 f8ff d0")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100E
	assert site.candidates == frozenset({Address(0x3000)})


def test_x86_stack_copy_survives_an_intervening_call() -> None:
	code = bytes.fromhex("5548 89 e548 c7 c3 00 30 00 0048 89 5d f8e8 00 00 00 0048 8b 45 f8ff d0")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1018
	assert site.candidates == frozenset({Address(0x3000)})


def test_x86_frame_slot_aliases_between_rbp_and_rsp() -> None:
	code = bytes.fromhex("5548 89 e548 c7 c3 00 30 00 0048 89 5d f848 8b 44 24 f8ff d0")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1014
	assert site.candidates == frozenset({Address(0x3000)})


def test_x86_push_pop_round_trip_carries_the_value() -> None:
	code = bytes.fromhex("48 c7 c3 00 30 00 005358ff d0")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.candidates == frozenset({Address(0x3000)})


def test_x86_push_immediate_pop_round_trip_carries_the_value() -> None:
	code = bytes.fromhex("68 00 30 00 0058ff d0")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.candidates == frozenset({Address(0x3000)})


def test_x86_sub_rsp_keeps_frame_offsets_consistent() -> None:
	code = bytes.fromhex("48 c7 c3 00 30 00 0048 83 ec 1048 89 1c 2448 8b 04 24ff d0")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.candidates == frozenset({Address(0x3000)})


def test_x86_indexed_stack_read_with_a_known_index() -> None:
	code = bytes.fromhex(
		"5548 89 e548 c7 c3 00 30 00 0048 89 5d f848 c7 c3 00 40 00 0048 89 5d f0"
		"48 c7 c1 00 00 00 0048 8b 44 cd f8ff d0"
	)
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)), ("first", 0x3000, 1), ("second", 0x4000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.candidates == frozenset({Address(0x3000)})


def test_x86_indexed_stack_read_with_a_top_index_is_unresolved() -> None:
	code = bytes.fromhex(
		"5548 89 e548 c7 c3 00 30 00 0048 89 5d f848 c7 c3 00 40 00 0048 89 5d f0"
		"48 8b 44 cd f8ff d0"
	)
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)), ("first", 0x3000, 1), ("second", 0x4000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.candidates == frozenset()


def test_x86_inc_shifts_the_tracked_address() -> None:
	code = bytes.fromhex("48 c7 c0 ff 2f 00 0048 ff c0ff d0")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.candidates == frozenset({Address(0x3000)})


def test_x86_xor_self_yields_zero_and_does_not_resolve() -> None:
	code = bytes.fromhex("48 31 c0ff d0")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)),),
	)
	(site,) = extract_call_sites(program)
	assert site.slot == 0
	assert site.candidates == frozenset({Address(0)})
	assert call_site_candidates(program, site, {}) == frozenset()


def test_x86_lea_frame_offset_then_memory_site_resolves() -> None:
	code = bytes.fromhex("5548 89 e548 c7 c3 00 30 00 0048 89 5d f848 8d 45 f8ff 10")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1013
	assert site.candidates == frozenset({Address(0x3000)})


def test_x86_32_frame_relative_memory_site_has_no_slot_address() -> None:
	code = bytes.fromhex("ff 55 08")
	program = _program(
		"EM_386",
		code,
		functions=(("caller", 0x1000, len(code)),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1000
	assert site.slot is None
	assert site.candidates == frozenset()


def test_arm_post_indexed_load_advances_the_base() -> None:
	body = bytes.fromhex("42 f2 00 0454 f8 04 3b98 47")
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x3000, 4)),
		objects=(("slot", 0x2000, _pointer(0x3000, 4)),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1008
	assert site.candidates == frozenset({Address(0x3000)})


def test_arm_push_then_sp_post_indexed_load_resolves() -> None:
	body = bytes.fromhex("43 f2 00 0410 b45d f8 04 3b98 47")
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x3000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100A
	assert site.candidates == frozenset({Address(0x3000)})


def test_arm_mov_from_sp_then_load_through_it_resolves() -> None:
	body = bytes.fromhex("43 f2 00 0410 b468 4601 6888 47")
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x3000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100A
	assert site.candidates == frozenset({Address(0x3000)})


def test_arm_scaled_add_with_top_index_enumerates_the_object() -> None:
	body = bytes.fromhex("02 4a02 eb c3 0149 6888 47") + b"\x00\xbf" + _pointer(0x2000, 4)
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("second", 0x4000, 4), ("fourth", 0x6000, 4)),
		objects=(
			(
				"table",
				0x2000,
				_pointer(0x3000, 4)
				+ _pointer(0x4000, 4)
				+ _pointer(0x5000, 4)
				+ _pointer(0x6000, 4),
			),
		),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1008
	assert site.candidates == frozenset({Address(0x4000), Address(0x6000)})


def test_arm_non_lsl_shifted_add_tops_the_destination() -> None:
	body = bytes.fromhex("02 4a02 eb d3 0149 6888 47") + b"\x00\xbf" * 2 + _pointer(0x2000, 4)
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)),),
		objects=(("table", 0x2000, _pointer(0x3000, 4) + _pointer(0x4000, 4)),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1008
	assert site.candidates == frozenset()


def test_arm_three_operand_immediate_add_shifts_the_value() -> None:
	body = bytes.fromhex("03 4b03 f1 04 0188 47") + b"\x00\xbf" * 4 + _pointer(0x2FFC, 4)
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x3000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1006
	assert site.candidates == frozenset({Address(0x3000)})


def test_x86_indexed_load_with_a_known_index_resolves() -> None:
	code = bytes.fromhex("48 c7 c3 00 20 00 0048 c7 c1 01 00 00 0048 8b 04 cbff d0")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)), ("first", 0x3000, 1), ("second", 0x4000, 1)),
		objects=(("table", 0x2000, _pointer(0x3000, 8) + _pointer(0x4000, 8)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1012
	assert site.candidates == frozenset({Address(0x4000)})


def test_x86_store_of_an_unknown_value_tops_the_slot() -> None:
	code = bytes.fromhex("5548 89 e548 89 4d f848 8b 45 f8ff d0")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100C
	assert site.candidates == frozenset()


def test_x86_push_of_a_memory_operand_resolves() -> None:
	code = bytes.fromhex("48 c7 c0 00 20 00 00ff 3059ff d1")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
		objects=(("slot", 0x2000, _pointer(0x3000, 8)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100A
	assert site.candidates == frozenset({Address(0x3000)})


def test_x86_add_to_a_memory_operand_writes_no_registers() -> None:
	code = bytes.fromhex("48 83 05 f9 0f 00 00 04ff d0")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1008
	assert site.candidates == frozenset()


def test_x86_inc_of_a_memory_operand_writes_no_registers() -> None:
	code = bytes.fromhex("48 ff 00ff d0")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1003
	assert site.candidates == frozenset()


def test_arm_str_then_reload_from_the_frame_resolves() -> None:
	body = bytes.fromhex("82 b043 f2 00 0001 9001 9988 47")
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x3000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100A
	assert site.candidates == frozenset({Address(0x3000)})


def test_arm_add_from_sp_then_load_through_it_resolves() -> None:
	body = bytes.fromhex("43 f2 00 0410 b400 a909 6888 47")
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x3000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100A
	assert site.candidates == frozenset({Address(0x3000)})


def test_arm_pop_round_trip_carries_the_value() -> None:
	body = bytes.fromhex("43 f2 00 0410 b402 bc88 47")
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x3000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1008
	assert site.candidates == frozenset({Address(0x3000)})


def test_arm_unshifted_register_add_uses_scale_one() -> None:
	body = bytes.fromhex("42 f2 00 0200 2302 eb 03 0109 6888 47")
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x3000, 4)),
		objects=(("slot", 0x2000, _pointer(0x3000, 4)),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100C
	assert site.candidates == frozenset({Address(0x3000)})


def test_x86_known_index_with_a_top_base_is_unresolved() -> None:
	code = bytes.fromhex("48 c7 c1 01 00 00 0048 8b 04 cbff d0")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100B
	assert site.candidates == frozenset()


def test_x86_stack_slot_write_union_overflows_k_to_top() -> None:
	table_one = b"".join((0x3000 + index * 8).to_bytes(8, "little") for index in range(33))
	table_two = b"".join((0x5000 + index * 8).to_bytes(8, "little") for index in range(32))
	code = bytes.fromhex(
		"5548 89 e548 c7 c3 00 20 00 0048 8b 04 cb48 89 45 f848 c7 c3 00 40 00 00"
		"48 8b 04 cb48 89 45 f848 8b 45 f8ff d0"
	)
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)),),
		objects=(("table_one", 0x2000, table_one), ("table_two", 0x4000, table_two)),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1026
	assert site.candidates == frozenset()


def test_x86_global_store_is_a_no_op_in_this_pass() -> None:
	code = bytes.fromhex("48 c7 c3 00 30 00 0048 89 0d f9 0f 00 00ff d3")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100E
	assert site.candidates == frozenset({Address(0x3000)})


def test_x86_xor_of_distinct_registers_tops_the_destination() -> None:
	code = bytes.fromhex("48 c7 c0 00 30 00 0048 31 d8ff d0")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100A
	assert site.candidates == frozenset()


def test_arm_pointer_store_is_a_no_op_in_this_pass() -> None:
	body = bytes.fromhex("43 f2 00 0242 f2 00 0108 6090 47")
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x3000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100A
	assert site.candidates == frozenset({Address(0x3000)})


def test_arm_post_indexed_store_advances_the_base() -> None:
	body = bytes.fromhex("42 f2 00 0143 f2 00 0041 f8 04 0b0a 6890 47")
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("second", 0x4000, 4)),
		objects=(("slot", 0x2004, _pointer(0x4000, 4)),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100E
	assert site.candidates == frozenset({Address(0x4000)})


def _multi_program(
	machine: str,
	code_by_address: dict[int, bytes],
	functions: tuple[tuple[str, int, int], ...],
	objects: tuple[tuple[str, int, bytes], ...] = (),
	pointer_size: int = 8,
) -> Program:
	return Program(
		byte_order="little",
		pointer_size=pointer_size,
		machine=machine,
		functions={
			Address(address): Function(
				name=name, address=Address(address), size=size, signature=None
			)
			for name, address, size in functions
		},
		objects={
			Address(address): DataObject(
				name=name, address=Address(address), size=len(data), type_name=None, signature=None
			)
			for name, address, data in objects
		},
		layouts={},
		relocations=(),
		sections={Address(address): code for address, code in code_by_address.items()}
		| {Address(address): data for _, address, data in objects},
	)


def test_x86_64_callee_is_seeded_from_the_caller_register_argument() -> None:
	program = _multi_program(
		"EM_X86_64",
		{
			0x1000: bytes.fromhex("48 c7 c7 00 30 00 00e8 f4 0f 00 00"),
			0x2000: bytes.fromhex("ff d7"),
		},
		functions=(("caller", 0x1000, 12), ("callee", 0x2000, 2), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x2000
	assert site.candidates == frozenset({Address(0x3000)})


def test_x86_32_callee_is_seeded_from_the_callers_first_stack_argument() -> None:
	program = _multi_program(
		"EM_386",
		{
			0x1000: bytes.fromhex("68 00 30 00 00e8 f6 0f 00 00"),
			0x2000: bytes.fromhex("5589 e58b 45 08ff d0"),
		},
		functions=(("caller", 0x1000, 10), ("callee", 0x2000, 8), ("target", 0x3000, 1)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x2000
	assert site.site_address == 0x2006
	assert site.candidates == frozenset({Address(0x3000)})


def test_x86_32_callee_is_seeded_from_the_callers_second_stack_argument() -> None:
	program = _multi_program(
		"EM_386",
		{
			0x1000: bytes.fromhex("68 00 40 00 0068 00 30 00 00e8 f1 0f 00 00"),
			0x2000: bytes.fromhex("5589 e58b 45 0cff d0"),
		},
		functions=(("caller", 0x1000, 15), ("callee", 0x2000, 8), ("second", 0x4000, 1)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x2000
	assert site.site_address == 0x2006
	assert site.candidates == frozenset({Address(0x4000)})


def test_arm_callee_is_seeded_from_the_caller_register_argument() -> None:
	program = _multi_program(
		"EM_ARM",
		{
			0x1000: bytes.fromhex("43 f2 00 0000 f0 fc ff"),
			0x2000: bytes.fromhex("80 47"),
		},
		functions=(("caller", 0x1000, 8), ("callee", 0x2000, 2), ("target", 0x3000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x2000
	assert site.candidates == frozenset({Address(0x3000)})


def test_seeds_propagate_along_a_call_chain() -> None:
	program = _multi_program(
		"EM_X86_64",
		{
			0x1000: bytes.fromhex("48 c7 c7 00 30 00 00e8 f4 0f 00 00"),
			0x2000: bytes.fromhex("e8 fb 1f 00 00"),
			0x4000: bytes.fromhex("ff d7"),
		},
		functions=(
			("first", 0x1000, 12),
			("middle", 0x2000, 5),
			("leaf", 0x4000, 2),
			("target", 0x3000, 1),
		),
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x4000
	assert site.candidates == frozenset({Address(0x3000)})


def test_callee_seed_joins_across_callers() -> None:
	program = _multi_program(
		"EM_X86_64",
		{
			0x1000: bytes.fromhex("48 c7 c7 00 30 00 00e8 f4 0f 00 00"),
			0x1200: bytes.fromhex("48 c7 c7 00 40 00 00e8 f4 0d 00 00"),
			0x2000: bytes.fromhex("ff d7"),
		},
		functions=(
			("first_caller", 0x1000, 12),
			("second_caller", 0x1200, 12),
			("callee", 0x2000, 2),
			("first", 0x3000, 1),
			("second", 0x4000, 1),
		),
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x2000
	assert site.candidates == frozenset({Address(0x3000), Address(0x4000)})


def test_cross_function_global_write_propagates_between_rounds() -> None:
	program = _multi_program(
		"EM_X86_64",
		{
			0x1000: bytes.fromhex("48 c7 05 f5 3f 00 0000 30 00 00c3"),
			0x2000: bytes.fromhex("48 8b 05 f9 2f 00 00ff d0"),
		},
		functions=(("writer", 0x1000, 12), ("reader", 0x2000, 9), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x2000
	assert site.site_address == 0x2007
	assert site.candidates == frozenset({Address(0x3000)})


def test_global_write_chain_caps_at_three_rounds() -> None:
	program = _multi_program(
		"EM_X86_64",
		{
			0x1000: bytes.fromhex("48 c7 05 f5 0f 00 0000 30 00 00c3"),
			0x1100: bytes.fromhex("48 8b 05 f9 0e 00 0048 89 05 02 0f 00 00c3"),
			0x1200: bytes.fromhex("48 8b 05 09 0e 00 0048 89 05 12 0e 00 00c3"),
			0x1300: bytes.fromhex("48 8b 05 19 0d 00 0048 89 05 22 0d 00 00c3"),
			0x1400: bytes.fromhex("48 8b 05 29 0c 00 0048 89 05 32 0c 00 00c3"),
			0x1500: bytes.fromhex("48 8b 05 39 0b 00 00ff d0"),
		},
		functions=(
			("first", 0x1000, 12),
			("second", 0x1100, 16),
			("third", 0x1200, 16),
			("fourth", 0x1300, 16),
			("fifth", 0x1400, 16),
			("reader", 0x1500, 9),
			("target", 0x3000, 1),
		),
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x1500
	assert site.candidates == frozenset()


def test_global_write_union_overflow_is_top() -> None:
	table_one = b"".join((0x3000 + index * 8).to_bytes(8, "little") for index in range(33))
	table_two = b"".join((0x5000 + index * 8).to_bytes(8, "little") for index in range(33))
	program = _multi_program(
		"EM_X86_64",
		{
			0x1000: bytes.fromhex("48 c7 c3 00 60 00 0048 8b 04 cb48 89 05 ee 0f 00 00c3"),
			0x1100: bytes.fromhex("48 c7 c3 00 70 00 0048 8b 04 cb48 89 05 ee 0e 00 00c3"),
			0x1200: bytes.fromhex("48 8b 05 f9 0d 00 00ff d0"),
		},
		functions=(
			("first_writer", 0x1000, 19),
			("second_writer", 0x1100, 19),
			("reader", 0x1200, 9),
		),
		objects=(("table_one", 0x6000, table_one), ("table_two", 0x7000, table_two)),
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x1200
	assert site.candidates == frozenset()


def test_recursive_seeding_converges() -> None:
	program = _multi_program(
		"EM_X86_64",
		{0x1000: bytes.fromhex("e8 fb ff ff ffff d7")},
		functions=(("recursive", 0x1000, 7),),
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x1000
	assert site.site_address == 0x1005
	assert site.candidates == frozenset()


def test_x86_call_arguments_with_a_top_stack_pointer_are_top() -> None:
	code = bytes.fromhex("83 e4 f0e8 00 00 00 00ff d0")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1008
	assert site.candidates == frozenset()


def test_x86_32_callee_seed_joins_across_callers() -> None:
	program = _multi_program(
		"EM_386",
		{
			0x1000: bytes.fromhex("68 00 30 00 00e8 f6 0f 00 00"),
			0x1100: bytes.fromhex("68 00 40 00 00e8 f6 0e 00 00"),
			0x2000: bytes.fromhex("5589 e58b 45 08ff d0"),
		},
		functions=(
			("first_caller", 0x1000, 10),
			("second_caller", 0x1100, 10),
			("callee", 0x2000, 8),
			("first", 0x3000, 1),
			("second", 0x4000, 1),
		),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x2000
	assert site.candidates == frozenset({Address(0x3000), Address(0x4000)})


def test_x86_64_callee_seed_overflow_is_top() -> None:
	table_one = b"".join((0x3000 + index * 8).to_bytes(8, "little") for index in range(33))
	table_two = b"".join((0x5000 + index * 8).to_bytes(8, "little") for index in range(33))
	program = _multi_program(
		"EM_X86_64",
		{
			0x1000: bytes.fromhex("48 c7 c3 00 60 00 0048 8b 3c cbe8 f0 0f 00 00"),
			0x1100: bytes.fromhex("48 c7 c3 00 70 00 0048 8b 3c cbe8 f0 0e 00 00"),
			0x2000: bytes.fromhex("ff d7"),
		},
		functions=(
			("first_caller", 0x1000, 16),
			("second_caller", 0x1100, 16),
			("callee", 0x2000, 2),
		),
		objects=(("table_one", 0x6000, table_one), ("table_two", 0x7000, table_two)),
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x2000
	assert len(site.candidates) == 33


def test_x86_32_callee_seed_overflow_is_top() -> None:
	table_one = b"".join((0x3000 + index * 4).to_bytes(4, "little") for index in range(33))
	table_two = b"".join((0x5000 + index * 4).to_bytes(4, "little") for index in range(33))
	program = _multi_program(
		"EM_386",
		{
			0x1000: bytes.fromhex("bb 00 60 00 008b 04 8b50e8 f2 0f 00 00"),
			0x1100: bytes.fromhex("bb 00 70 00 008b 04 8b50e8 f2 0e 00 00"),
			0x2000: bytes.fromhex("5589 e58b 45 08ff d0"),
		},
		functions=(
			("first_caller", 0x1000, 14),
			("second_caller", 0x1100, 14),
			("callee", 0x2000, 8),
		),
		objects=(("table_one", 0x6000, table_one), ("table_two", 0x7000, table_two)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x2000
	assert len(site.candidates) == 33


def test_a_global_write_feeding_a_call_argument_grows_the_seed_late() -> None:
	program = _multi_program(
		"EM_X86_64",
		{
			0x1000: bytes.fromhex("48 c7 05 f5 3f 00 0000 30 00 00c3"),
			0x1100: bytes.fromhex("48 8b 3d f9 3e 00 00e8 f4 0e 00 00"),
			0x2000: bytes.fromhex("ff d7"),
		},
		functions=(
			("writer", 0x1000, 12),
			("caller", 0x1100, 12),
			("callee", 0x2000, 2),
			("target", 0x3000, 1),
		),
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x2000
	assert site.candidates == frozenset({Address(0x3000)})


def test_the_round_budget_exhausts_on_a_deep_chain() -> None:
	program = _multi_program(
		"EM_X86_64",
		{
			0x1000: bytes.fromhex("48 c7 c7 00 30 00 0048 89 3d f2 0f 00 00e8 ed 00 00 00"),
			0x1100: bytes.fromhex("48 89 3d 09 0f 00 00e8 f4 00 00 00"),
			0x1200: bytes.fromhex("48 89 3d 19 0e 00 00e8 f4 00 00 00"),
			0x1300: bytes.fromhex("48 89 3d 29 0d 00 00e8 f4 00 00 00"),
			0x1400: bytes.fromhex("48 89 3d 39 0c 00 00e8 f4 00 00 00"),
			0x1500: bytes.fromhex("48 89 3d 49 0b 00 00e8 f4 00 00 00"),
			0x1600: bytes.fromhex("48 89 3d 59 0a 00 00e8 f4 00 00 00"),
			0x1700: bytes.fromhex("48 89 3d 69 09 00 00c3"),
			0x1800: bytes.fromhex("48 8b 05 79 08 00 00ff d0"),
		},
		functions=(
			("first", 0x1000, 19),
			("second", 0x1100, 12),
			("third", 0x1200, 12),
			("fourth", 0x1300, 12),
			("fifth", 0x1400, 12),
			("sixth", 0x1500, 12),
			("seventh", 0x1600, 12),
			("eighth", 0x1700, 8),
			("reader", 0x1800, 9),
			("target", 0x3000, 1),
		),
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x1800
	assert site.candidates == frozenset()


def test_x86_32_seed_adopts_arguments_from_different_callers() -> None:
	program = _multi_program(
		"EM_386",
		{
			0x1000: bytes.fromhex("5368 00 30 00 00e8 f5 0f 00 00"),
			0x1100: bytes.fromhex("68 00 40 00 0053e8 f5 0e 00 00"),
			0x2000: bytes.fromhex("5589 e58b 45 08ff d08b 45 0cff d0"),
		},
		functions=(
			("first_caller", 0x1000, 11),
			("second_caller", 0x1100, 11),
			("callee", 0x2000, 13),
			("first", 0x3000, 1),
			("second", 0x4000, 1),
		),
		pointer_size=4,
	)
	first_site, second_site = extract_call_sites(program)
	assert first_site.site_address == 0x2006
	assert first_site.candidates == frozenset({Address(0x3000)})
	assert second_site.site_address == 0x200B
	assert second_site.candidates == frozenset({Address(0x4000)})


def test_x86_global_store_then_load_resolves_within_the_function() -> None:
	code = bytes.fromhex("48 c7 c0 00 30 00 0048 89 05 f2 0f 00 0048 8b 05 eb 0f 00 00ff d0")
	program = _program(
		"EM_X86_64",
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1015
	assert site.candidates == frozenset({Address(0x3000)})


def _signature_program() -> Program:
	"""Three functions and one signature-carrying BSS slot for filter tests."""
	return Program(
		byte_order="little",
		pointer_size=8,
		machine="EM_X86_64",
		functions={
			Address(0x1000): Function(
				name="caller",
				address=Address(0x1000),
				size=12,
				signature=FunctionSignature(return_type="int", parameters=("int",)),
			),
			Address(0x3000): Function(
				name="matching_one",
				address=Address(0x3000),
				size=1,
				signature=FunctionSignature(return_type="void", parameters=("int",)),
			),
			Address(0x4000): Function(
				name="matching_two",
				address=Address(0x4000),
				size=1,
				signature=FunctionSignature(return_type="void", parameters=("int",)),
			),
			Address(0x5000): Function(
				name="non_matching",
				address=Address(0x5000),
				size=1,
				signature=FunctionSignature(return_type="int", parameters=()),
			),
		},
		objects={
			Address(0x2000): DataObject(
				name="bss_slot",
				address=Address(0x2000),
				size=8,
				type_name=None,
				signature=FunctionSignature(return_type="void", parameters=("int",)),
			)
		},
		layouts={},
		relocations=(),
		sections={Address(0x1000): bytes.fromhex("48 c7 c0 00 20 00 00ff d0")},
	)


def test_matching_targets_filters_by_exact_signature() -> None:
	program = _signature_program()
	signature = FunctionSignature(return_type="void", parameters=("int",))
	assert matching_targets(program, signature) == frozenset({Address(0x3000), Address(0x4000)})
	assert matching_targets(
		program, FunctionSignature(return_type="int", parameters=())
	) == frozenset({Address(0x5000)})
	assert (
		matching_targets(program, FunctionSignature(return_type="void", parameters=()))
		== frozenset()
	)


def test_chase_of_an_unreadable_slot_narrows_to_the_slot_signature() -> None:
	program = _signature_program()
	(site,) = extract_call_sites(program)
	signatures_by_slot = {
		Address(0x2000): FunctionSignature(return_type="void", parameters=("int",))
	}
	assert site.candidates == frozenset({Address(0x2000)})
	assert call_site_candidates(program, site, {}, signatures_by_slot) == frozenset(
		{Address(0x3000), Address(0x4000)}
	)


def test_unreadable_memory_site_narrows_to_its_slot_signature() -> None:
	program = Program(
		byte_order="little",
		pointer_size=8,
		machine="EM_X86_64",
		functions={
			Address(0x1000): Function(
				name="caller", address=Address(0x1000), size=6, signature=None
			),
			Address(0x3000): Function(
				name="matching",
				address=Address(0x3000),
				size=1,
				signature=FunctionSignature(return_type="void", parameters=("int",)),
			),
		},
		objects={
			Address(0x2000): DataObject(
				name="bss_slot",
				address=Address(0x2000),
				size=8,
				type_name=None,
				signature=FunctionSignature(return_type="void", parameters=("int",)),
			)
		},
		layouts={},
		relocations=(),
		sections={Address(0x1000): bytes.fromhex("ff 15 fa 0f 00 00")},
	)
	(site,) = extract_call_sites(program)
	assert site.slot == 0x2000
	assert site.candidates == frozenset()
	signatures_by_slot = {
		Address(0x2000): FunctionSignature(return_type="void", parameters=("int",))
	}
	assert call_site_candidates(program, site, {}, signatures_by_slot) == frozenset(
		{Address(0x3000)}
	)


def test_report_narrows_a_bss_site_to_its_slot_signature() -> None:
	program = Program(
		byte_order="little",
		pointer_size=8,
		machine="EM_X86_64",
		functions={
			Address(0x1000): Function(
				name="caller", address=Address(0x1000), size=6, signature=None
			),
			Address(0x3000): Function(
				name="matching",
				address=Address(0x3000),
				size=1,
				signature=FunctionSignature(return_type="void", parameters=("int",)),
			),
			Address(0x4000): Function(
				name="non_matching",
				address=Address(0x4000),
				size=1,
				signature=FunctionSignature(return_type="int", parameters=()),
			),
		},
		objects={
			Address(0x2000): DataObject(
				name="bss_slot",
				address=Address(0x2000),
				size=8,
				type_name="function pointer",
				signature=FunctionSignature(return_type="void", parameters=("int",)),
			)
		},
		layouts={},
		relocations=(),
		sections={Address(0x1000): bytes.fromhex("ff 15 fa 0f 00 00")},
	)
	sites = extract_call_sites(program)
	report = build_report(program, (), sites)
	(site_report,) = report.call_sites
	assert {candidate.name for candidate in site_report.candidates} == {"matching"}
