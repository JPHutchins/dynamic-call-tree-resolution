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
	Program,
	Provenance,
	SlotAssignment,
	assignments,
	call_site_candidates,
	extract_call_sites,
	load,
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


def test_x86_conditional_taken_path_to_the_site_conflicts_to_unresolved() -> None:
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


def test_x86_unmodeled_writes_are_cleared() -> None:
	program = _x86(
		bytes.fromhex("48 8b 05 f9 0f 00 0048 83 c0 04ff d0"),
		objects=(("slot", 0x2000, _pointer(0x3000, 8)),),
	)
	(site,) = extract_call_sites(program)
	assert site.slot is None


def test_x86_definitions_beyond_the_window_are_unknown() -> None:
	program = _x86(
		bytes.fromhex("48 b8 00 30 00 00 00 00 00 00") + b"\x90" * 12 + bytes.fromhex("ff d0")
	)
	(site,) = extract_call_sites(program)
	assert site.slot is None


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


def test_arm_movt_without_movw_keeps_only_the_high_half() -> None:
	body = bytes.fromhex("c5 f2 78 6398 47")
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1004
	assert site.slot == 0x56780000


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


def test_arm_unmodeled_writes_are_cleared() -> None:
	body = bytes.fromhex("01 4b 04 33 98 4700 bf") + _pointer(0x2000, 4)
	program = _program(
		"EM_ARM",
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x2000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1004
	assert site.slot is None


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
