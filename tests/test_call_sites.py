# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Tests for :mod:`dynamic_call_tree_resolution.call_sites`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from dynamic_call_tree_resolution import (
	Address,
	CallSite,
	DataObject,
	Function,
	FunctionSignature,
	Machine,
	Program,
	Provenance,
	Section,
	SlotAssignment,
	assignments,
	build_comparison,
	build_report,
	call_site_candidates,
	extract_call_sites,
	load,
	matching_targets,
	per_caller_candidates,
	resolve,
	signature_narrowings,
	unresolved_slots,
)
from dynamic_call_tree_resolution.model import FUNCTION_POINTER, InstructionSet, Unreached
from dynamic_call_tree_resolution.points_to import instruction_runs, instruction_set_at
from dynamic_call_tree_resolution.report import slot_counts
from dynamic_call_tree_resolution.stack_analysis import INDIRECT_CALLEE
from dynamic_call_tree_resolution.vsa import address_taken
from dynamic_call_tree_resolution.vsa.lattice import Known, Top
from tests.programs import build_program
from tests.sites import tracked_values
from tests.toolchains import FIXTURES, build_cortex_m3

if TYPE_CHECKING:
	from collections.abc import Collection, Mapping
	from pathlib import Path

	from dynamic_call_tree_resolution.vsa.lattice import ValueSet


def _program(
	machine: Machine,
	code: bytes,
	*,
	functions: tuple[tuple[str, int, int], ...] = (),
	objects: tuple[tuple[str, int, bytes], ...] = (),
	pointer_size: int = 8,
) -> Program:
	return build_program(
		machine,
		functions,
		objects=objects,
		sections={0x1000: code},
		pointer_size=pointer_size,
	)


def _pointer(value: int, size: int) -> bytes:
	return value.to_bytes(size, "little")


def _x86(code: bytes, *, objects: tuple[tuple[str, int, bytes], ...] = ()) -> Program:
	return _program(
		Machine.EM_X86_64,
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


def test_a_site_in_a_block_control_flow_never_reaches_is_unreached() -> None:
	(site,) = extract_call_sites(_x86(bytes.fromhex("c3 ff d0")))
	assert (site.site_address, site.target) == (0x1001, Unreached())


@pytest.mark.parametrize(
	("code", "loaded_from"),
	[
		pytest.param("48 8b 05 f9 0f 00 00ff d0", 0x2000, id="loaded"),
		pytest.param("48 8b 05 f9 0f 00 0048 89 c3ff d3", 0x2000, id="copied"),
		pytest.param("48 8b 05 f9 0f 00 0048 c7 c0 00 30 00 00ff d0", None, id="overwritten"),
		pytest.param(
			"48 8b 05 f9 0f 00 0048 85 c074 0748 8b 05 f5 0f 00 00ff d0",
			None,
			id="loaded from two slots on two paths",
		),
	],
)
def test_x86_register_site_records_the_one_slot_its_target_was_loaded_from(
	code: str, loaded_from: int | None
) -> None:
	(site,) = extract_call_sites(
		_x86(
			bytes.fromhex(code),
			objects=(
				("slot", 0x2000, _pointer(0x3000, 8)),
				("other", 0x2008, _pointer(0x3000, 8)),
			),
		)
	)
	assert site.loaded_from == loaded_from


@pytest.mark.parametrize(
	("machine", "code", "target"),
	[
		pytest.param(
			Machine.EM_X86_64,
			"b8 00 30 00 00ff d0",
			Known(values=frozenset({Address(0x3000)})),
			id="a 32-bit write zero-extends",
		),
		pytest.param(
			Machine.EM_X86_64,
			"48 c7 c0 00 30 00 00b8 00 20 00 00ff d0",
			Known(values=frozenset({Address(0x2000)})),
			id="a 32-bit write replaces the 64-bit value",
		),
		pytest.param(
			Machine.EM_X86_64,
			"48 c7 c0 00 30 00 00b0 01ff d0",
			Top(),
			id="an 8-bit write leaves the 64-bit value unknown",
		),
		pytest.param(
			Machine.EM_X86_64,
			"31 c0ff d0",
			Known(values=frozenset({Address(0)})),
			id="a 32-bit xor zeroes the 64-bit register",
		),
		pytest.param(
			Machine.EM_386,
			"b8 00 30 00 00b0 01ff d0",
			Top(),
			id="an 8-bit write leaves the 32-bit value unknown",
		),
	],
)
def test_an_x86_register_write_reaches_its_whole_register_family(
	machine: Machine, code: str, target: Known[Address] | Top
) -> None:
	(site,) = extract_call_sites(
		_program(
			machine,
			bytes.fromhex(code),
			functions=(("caller", 0x1000, len(bytes.fromhex(code))), ("target", 0x3000, 1)),
			pointer_size=8 if machine is Machine.EM_X86_64 else 4,
		)
	)
	assert site.target == target


def test_an_x86_64_32_bit_copy_of_a_frame_address_escapes_the_frame() -> None:
	code = bytes.fromhex("48 c7 c0 00 30 00 0050 8d 44 24 08e8 ef 1f 00 0058 ff d0")
	(site,) = extract_call_sites(
		build_program(
			Machine.EM_X86_64,
			(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
			sections={0x1000: code, 0x3000: b"\xc3"},
		)
	)
	assert site.target == Top()


def test_an_x86_8_bit_write_drops_the_slot_its_register_was_loaded_from() -> None:
	(site,) = extract_call_sites(
		_x86(
			bytes.fromhex("48 8b 05 f9 0f 00 00b0 01ff d0"),
			objects=(("slot", 0x2000, _pointer(0x3000, 8)),),
		)
	)
	assert (site.target, site.loaded_from) == (Top(), None)


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
	assert (site.slot, site.target) == (None, Top())
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
		Machine.EM_386,
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
		Machine.EM_386,
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
		Machine.EM_X86_64,
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
	assert site.target == Known(values=frozenset({Address(0x3000), Address(0x4000)}))


def test_conflicting_paths_fall_back_to_the_resolved_union() -> None:
	code = bytes.fromhex("48 c7 c3 00 30 00 0048 85 c074 0748 c7 c3 00 40 00 00ff d3")
	program = _program(
		Machine.EM_X86_64,
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
			provenance=Provenance.ROM_CONSTANT,
			relocated=False,
		),
		SlotAssignment(
			slot=Address(0x4000),
			path=("target_b",),
			candidates=frozenset({Address(0x4000)}),
			provenance=Provenance.ROM_CONSTANT,
			relocated=False,
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
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_arm_thumb_bit_twins_are_deduplicated() -> None:
	body = bytes.fromhex("00 4b 98 47") + _pointer(0x2000, 4)
	program = _program(
		Machine.EM_ARM,
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
		Machine.EM_ARM,
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
		Machine.EM_ARM,
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
		Machine.EM_ARM,
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
		Machine.EM_ARM,
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
		provenance=Provenance.ROM_CONSTANT,
		relocated=False,
	)
	assert call_site_candidates(program, site, {Address(0x3000): assignment}) == frozenset(
		{Address(0x2000)}
	)


def test_arm_bx_lr_is_a_return_not_a_site() -> None:
	program = _program(
		Machine.EM_ARM, bytes.fromhex("70 47"), functions=(("caller", 0x1000, 2),), pointer_size=4
	)
	assert extract_call_sites(program) == ()


def test_arm_bx_tail_site() -> None:
	body = bytes.fromhex("00 4b 18 47") + _pointer(0x2000, 4)
	program = _program(
		Machine.EM_ARM,
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
		Machine.EM_ARM,
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x2000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1006
	assert site.slot == 0x2000


def test_arm_stack_load_is_unresolved() -> None:
	program = _program(
		Machine.EM_ARM,
		bytes.fromhex("00 98 80 47"),
		functions=(("caller", 0x1000, 4),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.slot is None


def test_arm_load_with_untracked_base_is_unresolved() -> None:
	program = _program(
		Machine.EM_ARM,
		bytes.fromhex("18 68 80 47"),
		functions=(("caller", 0x1000, 4),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.slot is None


def test_arm_indexed_load_is_unresolved() -> None:
	program = _program(
		Machine.EM_ARM,
		bytes.fromhex("88 58 80 47"),
		functions=(("caller", 0x1000, 4),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.slot is None


def test_arm_callee_saved_register_survives_a_bl() -> None:
	body = bytes.fromhex("02 4c00 f0 00 f8a0 47") + b"\x00\xbf" * 2 + _pointer(0x3000, 4)
	program = _program(
		Machine.EM_ARM,
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
	body = bytes.fromhex("02 4c00 e000 24a0 47") + b"\x00\xbf" * 2 + _pointer(0x3000, 4)
	program = _program(
		Machine.EM_ARM,
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x2000, 4)),
		objects=(("slot", 0x3000, _pointer(0x2000, 4)),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1006
	assert site.slot == 0x3000
	assert call_site_candidates(program, site, {}) == frozenset({Address(0x2000)})


def test_arm_cbz_to_the_site_joins_the_taken_state() -> None:
	body = bytes.fromhex("02 4c00 b100 24a0 47") + b"\x00\xbf" * 2 + _pointer(0x3000, 4)
	program = _program(
		Machine.EM_ARM,
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x2000, 4)),
		objects=(("slot", 0x3000, _pointer(0x2000, 4)),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1006
	assert site.target == Known(values=frozenset({Address(0), Address(0x3000)}))


def test_arm_conditional_branch_past_the_site_keeps_fallthrough_state() -> None:
	body = bytes.fromhex("02 4ca4 4201 d0a0 47") + b"\x00\xbf" * 2 + _pointer(0x3000, 4)
	program = _program(
		Machine.EM_ARM,
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
		Machine.EM_ARM,
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
		Machine.EM_ARM,
		body,
		functions=(("caller", 0x1000, len(body)),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1004
	assert site.slot is None
	assert site.target == Top()


def test_arm_cbz_past_the_site_keeps_fallthrough_state() -> None:
	body = bytes.fromhex("02 4c0c b3a0 47") + b"\x00\xbf" * 3 + _pointer(0x3000, 4)
	program = _program(
		Machine.EM_ARM,
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x2000, 4)),
		objects=(("slot", 0x3000, _pointer(0x2000, 4)),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1004
	assert site.slot == 0x3000
	assert call_site_candidates(program, site, {}) == frozenset({Address(0x2000)})


def test_arm_empty_symbol_does_not_hide_its_sized_thumb_twin() -> None:
	body = bytes.fromhex("98 47 00 bf")
	program = _program(
		Machine.EM_ARM,
		body,
		functions=(("stub", 0x1000, 0), ("caller", 0x1001, len(body))),
		pointer_size=4,
	)
	assert [site.site_address for site in extract_call_sites(program)] == [0x1000]


@pytest.mark.image
@pytest.mark.parametrize(
	("elf", "register_indirect_branches"),
	[
		("hello/zephyr/zephyr.elf", 23),
		("sensor-two-impl/zephyr/zephyr.elf", 38),
	],
)
def test_arm_fixture_extracts_every_register_indirect_branch(
	zephyr_fixtures: Path, elf: str, register_indirect_branches: int
) -> None:
	assert len(extract_call_sites(load(zephyr_fixtures / elf))) == register_indirect_branches


def test_arm_call_in_window_clears_registers() -> None:
	body = bytes.fromhex("00 4b 00 f0 00 f8 98 47") + _pointer(0x2000, 4)
	program = _program(
		Machine.EM_ARM,
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
		Machine.EM_ARM,
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
		Machine.EM_ARM,
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
		Machine.EM_ARM,
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x2000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1002
	assert site.slot == 0x2000


def test_analyze_is_deterministic(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["nopie"])
	assert extract_call_sites(program) == extract_call_sites(program)


def test_a_target_loaded_from_a_slot_the_dynamic_loader_fills_is_unknown(
	fixture_elfs: dict[str, Path],
) -> None:
	program = load(fixture_elfs["nopie"])
	assert {
		(program.functions[site.caller_address].name, site.target, site.external)
		for site in extract_call_sites(program)
		if program.functions[site.caller_address].name in {"_init", "_start"}
	} == {("_init", Top(), "__gmon_start__"), ("_start", Top(), "__libc_start_main")}


def test_function_without_code_bytes_has_no_sites() -> None:
	program = _program(Machine.EM_X86_64, b"", functions=(("bare", 0x5000, 4),))
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


def test_per_caller_candidates_keeps_a_site_without_candidates_an_indirect_call(
	fixture_elfs: dict[str, Path],
) -> None:
	program = load(fixture_elfs["nopie"])
	resolved = assignments(program)
	by_caller, fallback = per_caller_candidates(program, extract_call_sites(program), resolved)
	assert set(by_caller) <= {function.name for function in program.functions.values()}
	assert INDIRECT_CALLEE in by_caller["main"]  # main's bss_cb site has no candidates
	assert fallback == {program.functions[address].name for address in address_taken(program)}
	assert {
		program.functions[address].name
		for assignment in resolved
		for address in assignment.candidates
	} < fallback


def test_x86_loop_merges_conditional_paths_into_a_union() -> None:
	code = bytes.fromhex("48 c7 c3 00 30 00 0048 85 c074 0a48 c7 c3 00 40 00 0048 ff c975 efff d3")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)), ("first", 0x3000, 1), ("second", 0x4000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1018
	assert site.target == Known(values=frozenset({Address(0x3000), Address(0x4000)}))
	assert call_site_candidates(program, site, {}) == frozenset({Address(0x3000), Address(0x4000)})


def test_x86_loop_carried_union_overflows_k_to_top() -> None:
	table = b"".join((0x3000 + index * 8).to_bytes(8, "little") for index in range(65))
	code = bytes.fromhex(
		"48 c7 c1 41 00 00 0048 c7 c3 00 20 00 0048 8b 0348 83 c3 0848 ff c975 f4ff d0"
	)
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)),),
		objects=(("table", 0x2000, table),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x101A
	assert site.target == Top()


def test_x86_top_index_over_baked_object_enumerates_its_slots() -> None:
	code = bytes.fromhex("48 c7 c3 00 20 00 0048 8b 04 cbff d0")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)), ("first", 0x3000, 1), ("second", 0x4000, 1)),
		objects=(("table", 0x2000, _pointer(0x3000, 8) + _pointer(0x4000, 8)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100B
	assert site.target == Known(values=frozenset({Address(0x3000), Address(0x4000)}))


def test_x86_top_index_without_an_enclosing_object_is_unresolved() -> None:
	code = bytes.fromhex("48 c7 c3 00 30 00 0048 8b 04 cbff d0")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100B
	assert site.target == Top()


def test_x86_top_index_over_a_too_small_object_is_unresolved() -> None:
	program = Program(
		byte_order="little",
		pointer_size=8,
		machine=Machine.EM_X86_64,
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
		data_in_code=(),
		arm_code=(),
		sections={
			Address(0x1000): Section(
				data=bytes.fromhex("48 c7 c3 00 20 00 0048 8b 04 cbff d0"), writable=False
			)
		},
	)
	(site,) = extract_call_sites(program)
	assert site.target == Top()


def _two_entry_table(objects: Mapping[Address, DataObject]) -> Program:
	return Program(
		byte_order="little",
		pointer_size=8,
		machine=Machine.EM_X86_64,
		functions={
			Address(0x1000): Function(
				name="caller", address=Address(0x1000), size=13, signature=None
			),
		},
		objects=objects,
		layouts={},
		relocations=(),
		data_in_code=(),
		arm_code=(),
		sections={
			Address(0x1000): Section(
				data=bytes.fromhex("48 c7 c3 00 20 00 0048 8b 04 cbff d0"), writable=False
			),
			Address(0x2000): Section(
				data=_pointer(0x3000, 8) + _pointer(0x4000, 8), writable=False
			),
		},
	)


def test_x86_top_index_ranges_over_the_enclosing_section() -> None:
	program = _two_entry_table(
		{
			Address(address): DataObject(
				name=name, address=Address(address), size=8, type_name=None, signature=None
			)
			for name, address in (("first_entry", 0x2000), ("second_entry", 0x2008))
		}
	)
	(site,) = extract_call_sites(program)
	assert site.target == Known(values=frozenset({Address(0x3000), Address(0x4000)}))


def test_x86_top_index_into_a_section_without_an_enclosing_object_is_unresolved() -> None:
	(site,) = extract_call_sites(_two_entry_table({}))
	assert site.target == Top()


def test_x86_bss_slot_read_is_unknown() -> None:
	program = Program(
		byte_order="little",
		pointer_size=8,
		machine=Machine.EM_X86_64,
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
		data_in_code=(),
		arm_code=(),
		sections={
			Address(0x1000): Section(
				data=bytes.fromhex("48 8b 05 f9 0f 00 00ff d0"), writable=False
			)
		},
	)
	(site,) = extract_call_sites(program)
	assert site.target == Top()


def test_x86_read_of_unmapped_memory_drops_the_value() -> None:
	code = bytes.fromhex("48 8b 05 f9 07 00 00ff d0")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)),),
	)
	(site,) = extract_call_sites(program)
	assert site.target == Known(values=frozenset[Address]())


def test_x86_join_with_one_path_missing_the_register_is_top() -> None:
	code = bytes.fromhex("48 c7 c0 00 30 00 0048 85 c074 0748 c7 c3 00 40 00 00eb 02ff d3")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1015
	assert site.target == Top()


def test_x86_join_with_the_other_path_missing_the_register_is_top() -> None:
	code = bytes.fromhex("48 c7 c0 00 30 00 0048 85 c075 0748 c7 c3 00 40 00 00eb 02ff d3")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1015
	assert site.target == Top()


def test_x86_loop_instruction_top_out_the_counter_and_keep_values() -> None:
	code = bytes.fromhex("b9 03 00 00 0048 c7 c0 00 30 00 00e2 f8ff d0")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100E
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_x86_stack_copy_survives_an_intervening_call() -> None:
	code = bytes.fromhex("5548 89 e548 c7 c3 00 30 00 0048 89 5d f8e8 00 00 00 0048 8b 45 f8ff d0")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1018
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_x86_frame_slot_aliases_between_rbp_and_rsp() -> None:
	code = bytes.fromhex("5548 89 e548 c7 c3 00 30 00 0048 89 5d f848 8b 44 24 f8ff d0")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1014
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_x86_push_pop_round_trip_carries_the_value() -> None:
	code = bytes.fromhex("48 c7 c3 00 30 00 005358ff d0")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_x86_push_immediate_pop_round_trip_carries_the_value() -> None:
	code = bytes.fromhex("68 00 30 00 0058ff d0")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_x86_sub_rsp_keeps_frame_offsets_consistent() -> None:
	code = bytes.fromhex("48 c7 c3 00 30 00 0048 83 ec 1048 89 1c 2448 8b 04 24ff d0")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_x86_indexed_stack_read_with_a_known_index() -> None:
	code = bytes.fromhex(
		"5548 89 e548 c7 c3 00 30 00 0048 89 5d f848 c7 c3 00 40 00 0048 89 5d f0"
		"48 c7 c1 00 00 00 0048 8b 44 cd f8ff d0"
	)
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)), ("first", 0x3000, 1), ("second", 0x4000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_x86_indexed_stack_read_with_a_top_index_is_unresolved() -> None:
	code = bytes.fromhex(
		"5548 89 e548 c7 c3 00 30 00 0048 89 5d f848 c7 c3 00 40 00 0048 89 5d f0"
		"48 8b 44 cd f8ff d0"
	)
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)), ("first", 0x3000, 1), ("second", 0x4000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.target == Top()


def test_x86_store_at_a_top_stack_index_clears_the_frame() -> None:
	code = bytes.fromhex("48 c7 c3 00 30 00 0048 89 5c 24 0848 89 04 cc48 8b 44 24 08ff d0")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.target == Top()


@pytest.mark.parametrize(
	"code",
	[
		pytest.param(
			"48c7c300300000 48895c2410 48895c2418 0f29442410 488b442418 ffd0",
			id="movaps-covers-two-slots",
		),
		pytest.param(
			"4885ff 7407 488d5c2408 eb05 488d5c2410 48c7c100300000 48890b 488b442410 ffd0",
			id="one-of-two-offsets-leaves-the-other-unknown",
		),
		pytest.param("48c7c300300000 48895c2408 488344240808 488b442408 ffd0", id="add-to-memory"),
		pytest.param(
			"48c7c300300000 48895c2408 c744240c00000000 488b442408 ffd0",
			id="a-narrower-store-overlaps-the-slot",
		),
		pytest.param(
			"48c7c300300000 48895c2408 488d7c2410 f348ab 488b442408 ffd0",
			id="rep-stosq-has-no-known-extent",
		),
		pytest.param(
			"48c7c300300000 48895c2408 0fae442410 488b442408 ffd0",
			id="fxsave-has-no-known-extent",
		),
	],
)
def test_x86_store_clobbers_the_frame_slots_it_may_cover(code: str) -> None:
	body = bytes.fromhex(code)
	program = _program(
		Machine.EM_X86_64,
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.target == Top()


@pytest.mark.parametrize(
	("code", "candidates"),
	[
		pytest.param(
			"488916 4885ff 7409 48c7c300400000 eb07 48c7c308400000 48c7c100500000 48890b"
			"488b042508400000 ffd0",
			set[int](),
			id="one-of-two-addresses-after-a-wild-store",
		),
		pytest.param("c604250140000000 488b042500400000 ffd0", set[int](), id="a-byte-in-the-slot"),
		pytest.param("48c7c300400000 8f03 488b03 ffd0", set[int](), id="pop-to-memory"),
		pytest.param("660f1f0400 488b042500400000 ffd0", {0x3000}, id="nop-writes-nothing"),
	],
)
def test_x86_store_to_writable_memory_clobbers_what_it_may_cover(
	code: str, candidates: set[int]
) -> None:
	body = bytes.fromhex(code)
	program = build_program(
		Machine.EM_X86_64,
		(("caller", 0x1000, len(body)), ("first", 0x3000, 1), ("second", 0x3100, 1)),
		objects=(("table", 0x4000, _pointer(0x3000, 8) + _pointer(0x3100, 8)),),
		sections={0x1000: body},
		writable=frozenset({0x4000}),
	)
	(site,) = extract_call_sites(program)
	assert site.target == _expected(candidates)


@pytest.mark.parametrize(
	("code", "candidates"),
	[
		pytest.param(
			"48c7c500400000 48c7c100310000 48894d00 488b042500400000 ffd0",
			{0x3000, 0x3100},
			id="store-through-a-known-rbp",
		),
		pytest.param("48c7c500400000 488b4508 ffd0", {0x3100}, id="load-through-a-known-rbp"),
		pytest.param("48c7c500400000 488d4508 488b00 ffd0", {0x3100}, id="lea-from-a-known-rbp"),
		pytest.param("48c7c500400000 4883c508 488b4500 ffd0", {0x3100}, id="add-to-a-known-rbp"),
		pytest.param("48c7c500400000 ff5508", {0x3100}, id="site-through-a-known-rbp"),
		pytest.param(
			"48894d00 488b042500400000 ffd0", set[int](), id="store-through-an-unknown-rbp"
		),
		pytest.param(
			"48c7c300300000 48895c2408 48894d00 488b442408 ffd0",
			set[int](),
			id="store-through-an-unknown-rbp-clears-the-frame",
		),
		pytest.param(
			"48c7c300300000 48895c2408 0fae4500 488b442408 ffd0",
			set[int](),
			id="fxsave-through-an-unknown-rbp-clears-the-frame",
		),
		pytest.param(
			"488d5c2408 4883c308 48c7c100310000 48894c2410 488b03 ffd0",
			{0x3100},
			id="add-to-an-sp-copy",
		),
	],
)
def test_x86_rbp_addresses_the_frame_only_as_an_sp_copy(code: str, candidates: set[int]) -> None:
	body = bytes.fromhex(code)
	program = build_program(
		Machine.EM_X86_64,
		(("caller", 0x1000, len(body)), ("first", 0x3000, 1), ("second", 0x3100, 1)),
		objects=(("table", 0x4000, _pointer(0x3000, 8) + _pointer(0x3100, 8)),),
		sections={0x1000: body},
		writable=frozenset({0x4000}),
	)
	(site,) = extract_call_sites(program)
	assert site.target == _expected(candidates)


def test_x86_inc_shifts_the_tracked_address() -> None:
	code = bytes.fromhex("48 c7 c0 ff 2f 00 0048 ff c0ff d0")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_x86_xor_self_yields_zero_and_does_not_resolve() -> None:
	code = bytes.fromhex("48 31 c0ff d0")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)),),
	)
	(site,) = extract_call_sites(program)
	assert site.slot == 0
	assert site.target == Known(values=frozenset({Address(0)}))
	assert call_site_candidates(program, site, {}) == frozenset()


def test_x86_lea_frame_offset_then_memory_site_resolves() -> None:
	code = bytes.fromhex("5548 89 e548 c7 c3 00 30 00 0048 89 5d f848 8d 45 f8ff 10")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1013
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_x86_32_frame_relative_memory_site_has_no_slot_address() -> None:
	code = bytes.fromhex("ff 55 08")
	program = _program(
		Machine.EM_386,
		code,
		functions=(("caller", 0x1000, len(code)),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1000
	assert site.slot is None
	assert site.target == Top()


def test_arm_post_indexed_load_advances_the_base() -> None:
	body = bytes.fromhex("42 f2 00 0454 f8 04 3b98 47")
	program = _program(
		Machine.EM_ARM,
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x3000, 4)),
		objects=(("slot", 0x2000, _pointer(0x3000, 4)),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1008
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_arm_push_then_sp_post_indexed_load_resolves() -> None:
	body = bytes.fromhex("43 f2 00 0410 b45d f8 04 3b98 47")
	program = _program(
		Machine.EM_ARM,
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x3000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100A
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_arm_mov_from_sp_then_load_through_it_resolves() -> None:
	body = bytes.fromhex("43 f2 00 0410 b468 4601 6888 47")
	program = _program(
		Machine.EM_ARM,
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x3000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100A
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_arm_scaled_add_with_top_index_enumerates_the_object() -> None:
	body = bytes.fromhex("02 4a02 eb c3 0149 6888 47") + b"\x00\xbf" + _pointer(0x2000, 4)
	program = _program(
		Machine.EM_ARM,
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
	assert site.target == Known(values=frozenset({Address(0x4000), Address(0x6000)}))


def test_arm_non_lsl_shifted_add_tops_the_destination() -> None:
	body = bytes.fromhex("02 4a02 eb d3 0149 6888 47") + b"\x00\xbf" * 2 + _pointer(0x2000, 4)
	program = _program(
		Machine.EM_ARM,
		body,
		functions=(("caller", 0x1000, len(body)),),
		objects=(("table", 0x2000, _pointer(0x3000, 4) + _pointer(0x4000, 4)),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1008
	assert site.target == Top()


def test_arm_three_operand_immediate_add_shifts_the_value() -> None:
	body = bytes.fromhex("03 4b03 f1 04 0188 47") + b"\x00\xbf" * 4 + _pointer(0x2FFC, 4)
	program = _program(
		Machine.EM_ARM,
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x3000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1006
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_x86_indexed_load_with_a_known_index_resolves() -> None:
	code = bytes.fromhex("48 c7 c3 00 20 00 0048 c7 c1 01 00 00 0048 8b 04 cbff d0")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)), ("first", 0x3000, 1), ("second", 0x4000, 1)),
		objects=(("table", 0x2000, _pointer(0x3000, 8) + _pointer(0x4000, 8)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1012
	assert site.target == Known(values=frozenset({Address(0x4000)}))


def test_x86_store_of_an_unknown_value_tops_the_slot() -> None:
	code = bytes.fromhex("5548 89 e548 89 4d f848 8b 45 f8ff d0")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100C
	assert site.target == Top()


def test_x86_push_of_a_memory_operand_resolves() -> None:
	code = bytes.fromhex("48 c7 c0 00 20 00 00ff 3059ff d1")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
		objects=(("slot", 0x2000, _pointer(0x3000, 8)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100A
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_x86_add_to_a_memory_operand_writes_no_registers() -> None:
	code = bytes.fromhex("48 83 05 f9 0f 00 00 04ff d0")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1008
	assert site.target == Top()


def test_x86_inc_of_a_memory_operand_writes_no_registers() -> None:
	code = bytes.fromhex("48 ff 00ff d0")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1003
	assert site.target == Top()


def test_arm_str_then_reload_from_the_frame_resolves() -> None:
	body = bytes.fromhex("82 b043 f2 00 0001 9001 9988 47")
	program = _program(
		Machine.EM_ARM,
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x3000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100A
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_arm_add_from_sp_then_load_through_it_resolves() -> None:
	body = bytes.fromhex("43 f2 00 0410 b400 a909 6888 47")
	program = _program(
		Machine.EM_ARM,
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x3000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100A
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_arm_pop_round_trip_carries_the_value() -> None:
	body = bytes.fromhex("43 f2 00 0410 b402 bc88 47")
	program = _program(
		Machine.EM_ARM,
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x3000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1008
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_arm_unshifted_register_add_uses_scale_one() -> None:
	body = bytes.fromhex("42 f2 00 0200 2302 eb 03 0109 6888 47")
	program = _program(
		Machine.EM_ARM,
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x3000, 4)),
		objects=(("slot", 0x2000, _pointer(0x3000, 4)),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100C
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_x86_known_index_with_a_top_base_is_unresolved() -> None:
	code = bytes.fromhex("48 c7 c1 01 00 00 0048 8b 04 cbff d0")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100B
	assert site.target == Top()


def test_x86_stack_slot_write_union_overflows_k_to_top() -> None:
	table_one = b"".join((0x3000 + index * 8).to_bytes(8, "little") for index in range(33))
	table_two = b"".join((0x5000 + index * 8).to_bytes(8, "little") for index in range(32))
	code = bytes.fromhex(
		"5548 89 e548 c7 c3 00 20 00 0048 8b 04 cb48 89 45 f848 c7 c3 00 40 00 00"
		"48 8b 04 cb48 89 45 f848 8b 45 f8ff d0"
	)
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)),),
		objects=(("table_one", 0x2000, table_one), ("table_two", 0x4000, table_two)),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1026
	assert site.target == Top()


def test_x86_global_store_is_a_no_op_in_this_pass() -> None:
	code = bytes.fromhex("48 c7 c3 00 30 00 0048 89 0d f9 0f 00 00ff d3")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100E
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_x86_xor_of_distinct_registers_tops_the_destination() -> None:
	code = bytes.fromhex("48 c7 c0 00 30 00 0048 31 d8ff d0")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100A
	assert site.target == Top()


def test_arm_pointer_store_is_a_no_op_in_this_pass() -> None:
	body = bytes.fromhex("43 f2 00 0242 f2 00 0108 6090 47")
	program = _program(
		Machine.EM_ARM,
		body,
		functions=(("caller", 0x1000, len(body)), ("target", 0x3000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100A
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_arm_post_indexed_store_advances_the_base() -> None:
	body = bytes.fromhex("42 f2 00 0143 f2 00 0041 f8 04 0b0a 6890 47")
	program = _program(
		Machine.EM_ARM,
		body,
		functions=(("caller", 0x1000, len(body)), ("second", 0x4000, 4)),
		objects=(("slot", 0x2004, _pointer(0x4000, 4)),),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x100E
	assert site.target == Known(values=frozenset({Address(0x4000)}))


def _multi_program(
	machine: Machine,
	code_by_address: dict[int, bytes],
	functions: tuple[tuple[str, int, int], ...],
	objects: tuple[tuple[str, int, bytes], ...] = (),
	pointer_size: int = 8,
) -> Program:
	return build_program(
		machine,
		functions,
		objects=objects,
		sections=code_by_address,
		pointer_size=pointer_size,
	)


def test_x86_64_callee_is_seeded_from_the_caller_register_argument() -> None:
	program = _multi_program(
		Machine.EM_X86_64,
		{
			0x1000: bytes.fromhex("48 c7 c7 00 30 00 00e8 f4 0f 00 00"),
			0x2000: bytes.fromhex("ff d7"),
		},
		functions=(("caller", 0x1000, 12), ("callee", 0x2000, 2), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x2000
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_x86_32_callee_is_seeded_from_the_callers_first_stack_argument() -> None:
	program = _multi_program(
		Machine.EM_386,
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
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_x86_32_callee_is_seeded_from_the_callers_second_stack_argument() -> None:
	program = _multi_program(
		Machine.EM_386,
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
	assert site.target == Known(values=frozenset({Address(0x4000)}))


def test_arm_callee_is_seeded_from_the_caller_register_argument() -> None:
	program = _multi_program(
		Machine.EM_ARM,
		{
			0x1000: bytes.fromhex("43 f2 00 0000 f0 fc ff"),
			0x2000: bytes.fromhex("80 47"),
		},
		functions=(("caller", 0x1000, 8), ("callee", 0x2000, 2), ("target", 0x3000, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x2000
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_arm_callee_known_only_by_its_thumb_symbol_is_seeded() -> None:
	program = _multi_program(
		Machine.EM_ARM,
		{
			0x1000: bytes.fromhex("43 f2 01 0000 f0 fc ff"),
			0x2000: bytes.fromhex("80 47"),
		},
		functions=(("caller", 0x1001, 8), ("callee", 0x2001, 2), ("target", 0x3001, 4)),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x2001
	assert site.target == Known(values=frozenset({Address(0x3001)}))


def test_seeds_propagate_along_a_call_chain() -> None:
	program = _multi_program(
		Machine.EM_X86_64,
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
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_callee_seed_joins_across_callers() -> None:
	program = _multi_program(
		Machine.EM_X86_64,
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
	assert site.target == Known(values=frozenset({Address(0x3000), Address(0x4000)}))


def test_cross_function_global_write_propagates_between_rounds() -> None:
	program = _multi_program(
		Machine.EM_X86_64,
		{
			0x1000: bytes.fromhex("48 c7 05 f5 3f 00 0000 30 00 00c3"),
			0x2000: bytes.fromhex("48 8b 05 f9 2f 00 00ff d0"),
		},
		functions=(("writer", 0x1000, 12), ("reader", 0x2000, 9), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x2000
	assert site.site_address == 0x2007
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_global_write_chain_propagates_to_a_fixpoint() -> None:
	program = _multi_program(
		Machine.EM_X86_64,
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
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_global_write_union_overflow_is_top() -> None:
	table_one = b"".join((0x3000 + index * 8).to_bytes(8, "little") for index in range(33))
	table_two = b"".join((0x5000 + index * 8).to_bytes(8, "little") for index in range(33))
	program = _multi_program(
		Machine.EM_X86_64,
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
	assert site.target == Top()


def test_recursive_seeding_converges() -> None:
	program = _multi_program(
		Machine.EM_X86_64,
		{0x1000: bytes.fromhex("e8 fb ff ff ffff d7")},
		functions=(("recursive", 0x1000, 7),),
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x1000
	assert site.site_address == 0x1005
	assert site.target == Top()


def test_x86_call_arguments_with_a_top_stack_pointer_are_top() -> None:
	code = bytes.fromhex("83 e4 f0e8 00 00 00 00ff d0")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)),),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1008
	assert site.target == Top()


def test_x86_32_callee_seed_joins_across_callers() -> None:
	program = _multi_program(
		Machine.EM_386,
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
	assert site.target == Known(values=frozenset({Address(0x3000), Address(0x4000)}))


def test_x86_64_callee_seed_overflow_is_top() -> None:
	table_one = b"".join((0x3000 + index * 8).to_bytes(8, "little") for index in range(33))
	table_two = b"".join((0x5000 + index * 8).to_bytes(8, "little") for index in range(33))
	program = _multi_program(
		Machine.EM_X86_64,
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
	assert site.target == Top()


def test_x86_32_callee_seed_overflow_is_top() -> None:
	table_one = b"".join((0x3000 + index * 4).to_bytes(4, "little") for index in range(33))
	table_two = b"".join((0x5000 + index * 4).to_bytes(4, "little") for index in range(33))
	program = _multi_program(
		Machine.EM_386,
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
	assert site.target == Top()


def test_a_global_write_feeding_a_call_argument_grows_the_seed_late() -> None:
	program = _multi_program(
		Machine.EM_X86_64,
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
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_the_round_budget_exhausts_on_a_deep_chain() -> None:
	program = _multi_program(
		Machine.EM_X86_64,
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
	assert site.target == Known(values=frozenset[Address]())


def test_x86_32_tail_jump_seeds_the_callee_from_the_arguments_above_the_return_address() -> None:
	program = _multi_program(
		Machine.EM_386,
		{
			0x1000: bytes.fromhex("68 00 30 00 00e8 f6 00 00 00c3"),
			0x1100: bytes.fromhex("e9 fb 00 00 00"),
			0x1200: bytes.fromhex("5589 e58b 45 08ff d05dc3"),
		},
		functions=(
			("main", 0x1000, 11),
			("forwarder", 0x1100, 5),
			("callee", 0x1200, 10),
			("target", 0x3000, 1),
		),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.caller_address == 0x1200
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def test_x86_32_seed_is_top_where_any_caller_passes_top() -> None:
	program = _multi_program(
		Machine.EM_386,
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
	assert (first_site.site_address, second_site.site_address) == (0x2006, 0x200B)
	assert (first_site.target, second_site.target) == (Top(), Top())


def test_x86_global_store_then_load_resolves_within_the_function() -> None:
	code = bytes.fromhex("48 c7 c0 00 30 00 0048 89 05 f2 0f 00 0048 8b 05 eb 0f 00 00ff d0")
	program = _program(
		Machine.EM_X86_64,
		code,
		functions=(("caller", 0x1000, len(code)), ("target", 0x3000, 1)),
	)
	(site,) = extract_call_sites(program)
	assert site.site_address == 0x1015
	assert site.target == Known(values=frozenset({Address(0x3000)}))


def _signature_program() -> Program:
	return Program(
		byte_order="little",
		pointer_size=8,
		machine=Machine.EM_X86_64,
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
		data_in_code=(),
		arm_code=(),
		sections={
			Address(0x1000): Section(
				data=bytes.fromhex("48 c7 c0 00 20 00 00ff d0"), writable=False
			),
			Address(0x6000): Section(data=(0x3000).to_bytes(8, "little"), writable=False),
		},
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


def test_chase_of_an_unreadable_slot_narrows_to_the_address_taken_functions_of_its_signature() -> (
	None
):
	program = _signature_program()
	(site,) = extract_call_sites(program)
	narrowings = signature_narrowings(
		program, {Address(0x2000): FunctionSignature(return_type="void", parameters=("int",))}
	)
	assert site.target == Known(values=frozenset({Address(0x2000)}))
	assert call_site_candidates(program, site, {}, narrowings) == frozenset({Address(0x3000)})


def test_unreadable_memory_site_narrows_to_its_slot_signature() -> None:
	program = Program(
		byte_order="little",
		pointer_size=8,
		machine=Machine.EM_X86_64,
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
		data_in_code=(),
		arm_code=(),
		sections={
			Address(0x1000): Section(data=bytes.fromhex("ff 15 fa 0f 00 00"), writable=False),
			Address(0x6000): Section(data=(0x3000).to_bytes(8, "little"), writable=False),
		},
	)
	(site,) = extract_call_sites(program)
	assert site.slot == 0x2000
	assert site.target == Top()
	narrowings = signature_narrowings(
		program, {Address(0x2000): FunctionSignature(return_type="void", parameters=("int",))}
	)
	assert call_site_candidates(program, site, {}, narrowings) == frozenset({Address(0x3000)})


def _bss_slot_site_program() -> Program:
	return Program(
		byte_order="little",
		pointer_size=8,
		machine=Machine.EM_X86_64,
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
			Address(0x5000): Function(
				name="never_stored",
				address=Address(0x5000),
				size=1,
				signature=FunctionSignature(return_type="void", parameters=("int",)),
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
		data_in_code=(),
		arm_code=(),
		sections={
			Address(0x1000): Section(data=bytes.fromhex("ff 15 fa 0f 00 00"), writable=False),
			Address(0x6000): Section(
				data=(0x4000).to_bytes(8, "little") + (0x3000).to_bytes(8, "little"),
				writable=False,
			),
		},
	)


@pytest.mark.parametrize(
	("narrow_by_signature", "expected"),
	[(False, frozenset[str]()), (True, frozenset({"matching"}))],
)
def test_report_narrows_a_bss_site_to_its_slot_signature_only_when_asked(
	narrow_by_signature: bool, expected: frozenset[str]
) -> None:
	program = _bss_slot_site_program()
	report = build_report(
		program, (), extract_call_sites(program), narrow_by_signature=narrow_by_signature
	)
	(site_report,) = report.call_sites
	assert frozenset(candidate.name for candidate in site_report.candidates) == expected


def test_stack_expansion_narrows_a_bss_site_to_its_slot_signature_only_when_asked() -> None:
	program = _bss_slot_site_program()
	sites = extract_call_sites(program)
	assert per_caller_candidates(program, sites, ()) == (
		{"caller": frozenset({INDIRECT_CALLEE})},
		frozenset({"matching", "non_matching"}),
	)
	assert per_caller_candidates(program, sites, (), narrow_by_signature=True) == (
		{"caller": frozenset({"matching"})},
		frozenset({"matching", "non_matching"}),
	)


def test_comparison_counts_a_narrowed_site_as_resolved_only_when_asked() -> None:
	program = _bss_slot_site_program()
	assert (
		build_comparison("bss.elf", program).resolved_call_sites,
		build_comparison("bss.elf", program, narrow_by_signature=True).resolved_call_sites,
	) == (0, 1)


@pytest.mark.parametrize(
	("machine", "code", "functions", "objects", "taken"),
	[
		pytest.param(
			Machine.EM_X86_64,
			"c3",
			(("caller", 0x1000, 1), ("target", 0x3000, 1)),
			(("slot", 0x2000, (0x3000).to_bytes(8, "little")),),
			{"target"},
			id="x86-64 data slot",
		),
		pytest.param(
			Machine.EM_386,
			"68 00 30 00 00",
			(("caller", 0x1000, 5), ("target", 0x3000, 1)),
			(),
			{"target"},
			id="x86 push imm32 stored in code",
		),
		pytest.param(
			Machine.EM_X86_64,
			"bf 00 30 00 00",
			(("caller", 0x1000, 5), ("target", 0x3000, 1)),
			(),
			{"target"},
			id="x86-64 mov imm32",
		),
		pytest.param(
			Machine.EM_X86_64,
			"48 8d 3d f9 1f 00 00",
			(("caller", 0x1000, 7), ("target", 0x3000, 1)),
			(),
			{"target"},
			id="x86-64 rip-relative lea",
		),
		pytest.param(
			Machine.EM_X86_64,
			"e8 fb 1f 00 00",
			(("caller", 0x1000, 5), ("target", 0x3000, 1)),
			(),
			set[str](),
			id="x86-64 call target",
		),
		pytest.param(
			Machine.EM_ARM,
			"40 f2 01 00 c0 f2 01 00",
			(("caller", 0x1001, 8), ("target", 0x10001, 2)),
			(),
			{"target"},
			id="thumb movw movt",
		),
		pytest.param(
			Machine.EM_ARM,
			"c0 f2 01 00",
			(("caller", 0x1001, 4), ("target", 0x10001, 2)),
			(),
			set[str](),
			id="thumb movt alone",
		),
		pytest.param(
			Machine.EM_ARM,
			"c4 bf 40 f2 01 00 c0 f2 01 00",
			(("caller", 0x1001, 10), ("target", 0x10001, 2)),
			(),
			{"target"},
			id="thumb predicated movw movt",
		),
		pytest.param(
			Machine.EM_ARM,
			"c8 bf 0f f2 04 00",
			(("caller", 0x1001, 6), ("target", 0x1009, 2)),
			(),
			{"target"},
			id="thumb predicated addw pc",
		),
		pytest.param(
			Machine.EM_ARM,
			"c8 bf 01 a0",
			(("caller", 0x1001, 4), ("target", 0x1009, 2)),
			(),
			{"target"},
			id="thumb predicated adr",
		),
		pytest.param(
			Machine.EM_ARM,
			"01 a0",
			(("caller", 0x1001, 2), ("target", 0x1009, 2)),
			(),
			{"target"},
			id="thumb adr",
		),
		pytest.param(
			Machine.EM_ARM,
			"0f f2 04 00",
			(("caller", 0x1001, 4), ("target", 0x1009, 2)),
			(),
			{"target"},
			id="thumb addw pc",
		),
		pytest.param(
			Machine.EM_ARM,
			"af f2 08 00",
			(("caller", 0x1001, 4), ("target", 0xFFD, 2)),
			(),
			{"target"},
			id="thumb subw pc",
		),
		pytest.param(
			Machine.EM_ARM,
			"00 f0 fe ff",
			(("caller", 0x1001, 4), ("target", 0x2001, 2)),
			(),
			set[str](),
			id="thumb bl target",
		),
	],
)
def test_address_taken_finds_stored_and_computed_function_addresses(
	machine: Machine,
	code: str,
	functions: tuple[tuple[str, int, int], ...],
	objects: tuple[tuple[str, int, bytes], ...],
	taken: set[str],
) -> None:
	program = _program(
		machine,
		bytes.fromhex(code),
		functions=functions,
		objects=objects,
		pointer_size=8 if machine == Machine.EM_X86_64 else 4,
	)
	assert {program.functions[address].name for address in address_taken(program)} == taken


def test_address_taken_finds_a_function_above_64_kib_only_a_predicated_movw_movt_pair_builds(
	tmp_path: Path,
) -> None:
	source = tmp_path / "far.c"
	source.write_text(
		"[[gnu::used, gnu::noinline]] void pad(void) {\n"
		'\t__asm__ volatile(".space 70000");\n'
		"}\n\n" + (FIXTURES / "reproducers" / "code_address_callback.c").read_text()
	)
	program = load(
		build_cortex_m3(
			(source,),
			tmp_path / "far.elf",
			"-O2",
			"-mslow-flash-data",
			"-fno-toplevel-reorder",
		)
	)
	(deep,) = (
		address for address, function in program.functions.items() if function.name == "deep"
	)
	assert (deep > 0xFFFF, program.link_references, deep in address_taken(program)) == (
		True,
		(),
		True,
	)


@pytest.mark.image
@pytest.mark.parametrize(
	"elf",
	[
		"hello/zephyr/zephyr.elf",
		"sensor-two-impl/zephyr/zephyr.elf",
		"counter-su/zephyr/zephyr.exe",
	],
)
def test_every_function_a_site_candidate_names_is_address_taken(
	zephyr_fixtures: Path, elf: str
) -> None:
	program = load(zephyr_fixtures / elf)
	taken = address_taken(program)
	assert [
		program.functions[address].name
		for site in extract_call_sites(program)
		for address in tracked_values(site)
		if address in program.functions and address not in taken
	] == []


@pytest.mark.parametrize(
	("machine", "code", "target", "candidates"),
	[
		pytest.param(
			Machine.EM_ARM,
			"42f20004 c0f20004 abbe a047",
			0x2000,
			{0x2000},
			id="thumb-bkpt-keeps-r4",
		),
		pytest.param(
			Machine.EM_ARM, "42f20004 c0f20004 03df a047", 0x2000, {0x2000}, id="thumb-svc-keeps-r4"
		),
		pytest.param(
			Machine.EM_ARM,
			"42f20000 c0f20000 abbe 8047",
			0x2000,
			set[int](),
			id="thumb-bkpt-clobbers-r0",
		),
		pytest.param(
			Machine.EM_ARM,
			"42f20004 c0f20004 00de a047",
			0x2000,
			set[int](),
			id="thumb-udf-is-terminal",
		),
		pytest.param(
			Machine.EM_X86_64, "48c7c300300000 cc ffd3", 0x3000, {0x3000}, id="x86-int3-keeps-rbx"
		),
		pytest.param(
			Machine.EM_X86_64,
			"48c7c300300000 0f05 ffd3",
			0x3000,
			{0x3000},
			id="x86-syscall-keeps-rbx",
		),
		pytest.param(
			Machine.EM_X86_64, "48c7c300300000 cd80 ffd3", 0x3000, {0x3000}, id="x86-int-keeps-rbx"
		),
		pytest.param(
			Machine.EM_X86_64, "48c7c300300000 f4 ffd3", 0x3000, {0x3000}, id="x86-hlt-keeps-rbx"
		),
		pytest.param(
			Machine.EM_X86_64,
			"48c7c000300000 cc ffd0",
			0x3000,
			set[int](),
			id="x86-int3-clobbers-rax",
		),
		pytest.param(
			Machine.EM_X86_64,
			"48c7c300300000 0f0b ffd3",
			0x3000,
			set[int](),
			id="x86-ud2-is-terminal",
		),
	],
)
def test_a_returning_trap_falls_through_and_clobbers_caller_saved_registers(
	machine: Machine, code: str, target: int, candidates: set[int]
) -> None:
	body = bytes.fromhex(code)
	program = _program(
		machine,
		body,
		functions=(("caller", 0x1000, len(body)), ("target", target, 4)),
		pointer_size=4 if machine == Machine.EM_ARM else 8,
	)
	(site,) = extract_call_sites(program)
	assert site.target == _expected(candidates)


@pytest.mark.parametrize(
	("code", "candidates"),
	[
		pytest.param(
			"42f20003 0193 43f20002 0128 c8bf 0192 019b 9847",
			{0x2000, 0x3000},
			id="strgt-joins-the-slot",
		),
		pytest.param(
			"42f20003 0128 c8bf 43f20003 9847", {0x2000, 0x3000}, id="movwgt-joins-the-register"
		),
		pytest.param("42f20002 43f20003 cde90223 039c a047", {0x3000}, id="strd-writes-two-words"),
		pytest.param(
			"6846 43f20001 45f20002 06c0 50f8043c 9847", {0x5000}, id="stm-writes-and-advances"
		),
		pytest.param("42f20003 0193 8df80530 019c a047", set[int](), id="strb-overlaps-the-slot"),
		pytest.param(
			"42f20003 4df80c3c 2ded020b 5df8044c a047", {0x2000}, id="vpush-moves-sp-past-the-slot"
		),
		pytest.param(
			"6846 42f20003 0093 a0ec020b 009c a047", set[int](), id="vstmia-covers-the-slot"
		),
		pytest.param("42f20002 6946 41e80032 9047", set[int](), id="strex-writes-its-status"),
		pytest.param("42f20003 0093 80ed0021 009c a047", set[int](), id="stc-has-no-known-extent"),
	],
)
def test_thumb_store_writes_the_words_it_covers(code: str, candidates: set[int]) -> None:
	body = bytes.fromhex(code)
	program = _program(
		Machine.EM_ARM,
		body,
		functions=(
			("caller", 0x1000, len(body)),
			("first", 0x2000, 4),
			("second", 0x3000, 4),
			("third", 0x5000, 4),
		),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.target == _expected(candidates)


def _thumb_slot_program(sections: Mapping[int, bytes]) -> Program:
	return build_program(
		Machine.EM_ARM,
		(
			*((f"code_{address:x}", address, len(code)) for address, code in sections.items()),
			("first", 0x2000, 4),
			("second", 0x3000, 4),
		),
		objects=(("slot", 0x4000, _pointer(0x2000, 4)),),
		sections=sections,
		writable=frozenset({0x4000}),
		pointer_size=4,
	)


def test_thumb_predicated_store_reaches_other_readers() -> None:
	program = _thumb_slot_program(
		{
			0x1000: bytes.fromhex("44f20001 43f20002 0128 c8bf 0a60 7047"),
			0x1100: bytes.fromhex("44f20001 0b68 9847"),
		}
	)
	(site,) = extract_call_sites(program)
	assert site.target == Known(values=frozenset({Address(0x2000), Address(0x3000)}))


def test_thumb_predicated_store_after_a_wild_store_is_unknown() -> None:
	program = _thumb_slot_program(
		{0x1000: bytes.fromhex("2c60 44f20001 43f20002 0128 c8bf 0a60 0b68 9847")}
	)
	(site,) = extract_call_sites(program)
	assert site.target == Top()


@pytest.mark.parametrize(
	("code", "callee", "candidates"),
	[
		pytest.param("42f20003019300f002f8019b98477047", 0xE, {0x2000}, id="kept-without-escape"),
		pytest.param(
			"42f20003 0193 6846 00f002f8 019b 9847 7047", 0x10, set[int](), id="passed-in-r0"
		),
		pytest.param(
			"6946 44f20002 1160 42f20003 0193 00f002f8 019b 9847 7047",
			0x16,
			set[int](),
			id="stored-to-a-global",
		),
		pytest.param(
			"0028 00d0 6946 42f20003 0193 00f002f8 019b 9847 7047",
			0x14,
			set[int](),
			id="lost-in-a-join",
		),
		pytest.param(
			"6c46 44f00100 42f20003 0193 00f002f8 019b 9847 7047",
			0x14,
			set[int](),
			id="derived-by-orr",
		),
		pytest.param(
			"6c46 2578 42f20003 0193 00f002f8 019b 9847 7047",
			0x12,
			{0x2000},
			id="kept-after-a-byte-load-through-it",
		),
		pytest.param(
			"0deb0100 42f20003 0193 00f002f8 019b 9847 7047",
			0x12,
			set[int](),
			id="indexed-by-add",
		),
		pytest.param(
			"6c46 201d 42f20003 0193 00f002f8 019b 9847 7047",
			0x12,
			set[int](),
			id="offset-by-adds-from-a-copy",
		),
		pytest.param(
			"6c46 a01e 42f20003 0193 00f002f8 019b 9847 7047",
			0x12,
			set[int](),
			id="offset-by-subs-from-a-copy",
		),
		pytest.param(
			"6c46 44f00100 42f20003 0193 2a60 019b 9847",
			0x12,
			set[int](),
			id="wild-store-after-an-escape",
		),
		pytest.param("42f20003 0193 2a60 019b 9847", 0xC, {0x2000}, id="wild-store-without-escape"),
	],
)
def test_thumb_frame_survives_calls_and_wild_stores_until_its_address_escapes(
	code: str, callee: int, candidates: set[int]
) -> None:
	body = bytes.fromhex(code)
	program = _program(
		Machine.EM_ARM,
		body,
		functions=(
			("caller", 0x1000, callee),
			*((("callee", 0x1000 + callee, len(body) - callee),) if callee < len(body) else ()),
			("first", 0x2000, 4),
		),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.target == _expected(candidates)


def test_thumb_call_forgets_the_writes_on_its_path() -> None:
	body = bytes.fromhex("44f20004 42f20003 2360 00f002f8 2368 9847 44f20001 43f20002 0a60 7047")
	program = build_program(
		Machine.EM_ARM,
		(
			("global_case", 0x1000, 0x12),
			("install", 0x1012, len(body) - 0x12),
			("a", 0x2000, 4),
			("b", 0x3000, 4),
			("baked", 0x5000, 4),
		),
		objects=(("handler", 0x4000, _pointer(0x5000, 4)),),
		sections={0x1000: body},
		writable=frozenset({0x4000}),
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.target == Known(
		values=frozenset({Address(0x2000), Address(0x3000), Address(0x5000)})
	)


def test_thumb_trap_leaves_the_writes_before_it_in_the_summary() -> None:
	program = _thumb_slot_program(
		{
			0x1000: bytes.fromhex("44f20001 43f20002 0a60 00df 7047"),
			0x1100: bytes.fromhex("44f20001 0b68 9847"),
		}
	)
	(site,) = extract_call_sites(program)
	assert site.target == Known(values=frozenset({Address(0x2000), Address(0x3000)}))


@pytest.mark.parametrize(
	("code", "callee"),
	[
		pytest.param(
			"48c7c100300000 48894c2408 488d7c2410 e807000000 488b442408 ffd0 c3",
			0x1D,
			id="passed-in-rdi",
		),
		pytest.param(
			"48c7c100300000 48894c2408 488d442410 50 e807000000 488b442410 ffd0 c3",
			0x1E,
			id="pushed",
		),
		pytest.param(
			"488d442410 48c7c300400000 480103 48c7c100300000 48894c2408 e807000000 488b442408"
			"ffd0 c3",
			0x27,
			id="added-to-memory",
		),
	],
)
def test_x86_call_clears_the_frame_after_its_address_escapes(code: str, callee: int) -> None:
	body = bytes.fromhex(code)
	program = _program(
		Machine.EM_X86_64,
		body,
		functions=(
			("caller", 0x1000, callee),
			("callee", 0x1000 + callee, len(body) - callee),
			("target", 0x3000, 1),
		),
	)
	(site,) = extract_call_sites(program)
	assert site.target == Top()


def _hook_program(machine: Machine, code: bytes, *, writable: bool, pointer_size: int) -> Program:
	return Program(
		byte_order="little",
		pointer_size=pointer_size,
		machine=Machine(machine),
		functions={
			Address(address): Function(
				name=name, address=Address(address), size=size, signature=None
			)
			for name, address, size in (
				("code", 0x1000, len(code)),
				("a", 0x2000, 4),
				("b", 0x3000, 4),
			)
		},
		objects={
			Address(0x4000): DataObject(
				name="hook",
				address=Address(0x4000),
				size=pointer_size,
				type_name=FUNCTION_POINTER,
				signature=None,
			)
		},
		layouts={},
		relocations=(),
		data_in_code=(),
		arm_code=(),
		sections={
			Address(0x1000): Section(data=code, writable=False),
			Address(0x4000): Section(data=_pointer(0x2000, pointer_size), writable=writable),
		},
	)


@pytest.mark.parametrize(
	("code", "candidates"),
	[
		pytest.param("44f20001 43f20002 0a60 7047", {0x2000, 0x3000}, id="a-stored-function"),
		pytest.param("44f20001 40f20002 0a60 7047", {0x2000}, id="a-stored-null"),
		pytest.param("44f20001 45f20002 0a60 7047", None, id="a-stored-non-function"),
	],
)
def test_ram_initializer_holds_what_the_program_stores_to_its_slot(
	code: str, candidates: set[int] | None
) -> None:
	program = _hook_program(Machine.EM_ARM, bytes.fromhex(code), writable=True, pointer_size=4)
	resolved = resolve(program).assignments
	assert {
		assignment.slot: (assignment.provenance, assignment.candidates) for assignment in resolved
	} == (
		{
			Address(0x4000): (
				Provenance.RAM_INITIALIZER,
				frozenset(Address(address) for address in candidates),
			)
		}
		if candidates is not None
		else {}
	)
	assert slot_counts(resolved, unresolved_slots(program, resolved)).total_slots == 1


@pytest.mark.parametrize(
	("writable", "candidates"),
	[
		pytest.param(True, set[int](), id="ram-initializer-is-unresolved"),
		pytest.param(False, {0x2000}, id="rom-constant-keeps-its-value"),
	],
)
def test_unknown_store_unresolves_only_writable_slots(writable: bool, candidates: set[int]) -> None:
	program = _hook_program(
		Machine.EM_ARM, bytes.fromhex("2c60 7047"), writable=writable, pointer_size=4
	)
	resolved = resolve(program).assignments
	assert {candidate for assignment in resolved for candidate in assignment.candidates} == {
		Address(address) for address in candidates
	}
	assert slot_counts(resolved, unresolved_slots(program, resolved)).total_slots == 1


@pytest.mark.parametrize(
	("writable", "candidates"),
	[
		pytest.param(True, set[int](), id="writable-slot-is-not-read"),
		pytest.param(False, {0x2000}, id="read-only-slot-is-read"),
	],
)
def test_chasing_a_slot_reads_its_image_value_only_when_read_only(
	writable: bool, candidates: set[int]
) -> None:
	program = _hook_program(
		Machine.EM_ARM, bytes.fromhex("7047"), writable=writable, pointer_size=4
	)
	site = CallSite(
		caller_address=Address(0x1000),
		site_address=Address(0x1000),
		slot=None,
		target=Known(values=frozenset({Address(0x4000)})),
	)
	assert call_site_candidates(program, site, {}) == frozenset(
		Address(address) for address in candidates
	)


def test_site_through_an_unresolved_slot_keeps_its_member_path() -> None:
	program = _hook_program(
		Machine.EM_X86_64, bytes.fromhex("488916 ff142500400000"), writable=True, pointer_size=8
	)
	resolution = resolve(program)
	(site,) = build_report(program, resolution.assignments, resolution.sites).call_sites
	assert site.member_path == "hook"
	assert site.candidates == ()


@pytest.mark.parametrize(
	("code", "callee", "sites"),
	[
		pytest.param("42f20003 0128 c8bf 9847 7047", None, [{0x2000}], id="blxgt-is-a-site"),
		pytest.param(
			"42f20003 43f20004 0028 18bf 1847 a047",
			None,
			[{0x2000}, {0x3000}],
			id="bxne-is-a-site-and-falls-through",
		),
		pytest.param(
			"42f20000 0129 c8bf 00f001f8 8047 7047", 0xE, [set[int]()], id="blgt-clobbers-r0"
		),
		pytest.param("42f20003 0028 18bf 7047 9847", None, [{0x2000}], id="bxne-lr-falls-through"),
		pytest.param(
			"10b5 42f20003 0028 18bf 10bd 9847", None, [{0x2000}], id="popne-pc-falls-through"
		),
	],
)
def test_thumb_predicated_transfer_also_falls_through(
	code: str, callee: int | None, sites: list[set[int]]
) -> None:
	body = bytes.fromhex(code)
	caller_size = callee if callee is not None else len(body)
	program = _program(
		Machine.EM_ARM,
		body,
		functions=(
			("caller", 0x1000, caller_size),
			*((("callee", 0x1000 + caller_size, len(body) - caller_size),) if callee else ()),
			("first", 0x2000, 4),
			("second", 0x3000, 4),
		),
		pointer_size=4,
	)
	assert [
		candidates
		for _, candidates in sorted(
			(site.site_address, site.target) for site in extract_call_sites(program)
		)
	] == [_expected(expected) for expected in sites]


@pytest.mark.parametrize(
	("code", "candidates"),
	[
		pytest.param("44f20004 54f8043f 6568 a847", {0x5000}, id="ldr-advances-its-base"),
		pytest.param("42f20003 4df8083d 009c a047", {0x2000}, id="str-advances-sp"),
	],
)
def test_thumb_pre_indexed_writeback_advances_the_base(code: str, candidates: set[int]) -> None:
	body = bytes.fromhex(code)
	program = build_program(
		Machine.EM_ARM,
		(("caller", 0x1000, len(body)), ("a", 0x2000, 4), ("b", 0x3000, 4), ("c", 0x5000, 4)),
		objects=(
			("table", 0x4000, _pointer(0x2000, 4) + _pointer(0x3000, 4) + _pointer(0x5000, 4)),
		),
		sections={0x1000: body},
		pointer_size=4,
	)
	(site,) = extract_call_sites(program)
	assert site.target == _expected(candidates)


@pytest.mark.parametrize(
	("data_in_code", "sites"),
	[
		pytest.param(((0x1008, 0x100C),), [{0x2000}], id="a-marked-literal-is-not-decoded"),
		pytest.param((), [{0x2000}, set[int]()], id="an-unmarked-literal-decodes-as-blx"),
	],
)
def test_thumb_data_in_code_is_not_decoded(
	data_in_code: tuple[tuple[int, int], ...], sites: list[set[int]]
) -> None:
	body = bytes.fromhex("42f20003 9847 7047 90470000")
	program = build_program(
		Machine.EM_ARM,
		(("caller", 0x1000, len(body)), ("first", 0x2000, 4)),
		sections={0x1000: body},
		pointer_size=4,
		data_in_code=data_in_code,
	)
	assert [
		candidates
		for _, candidates in sorted(
			(site.site_address, site.target) for site in extract_call_sites(program)
		)
	] == [_expected(expected) for expected in sites]


def _thumb_dispatch_program(code: str, data_in_code: tuple[tuple[int, int], ...] = ()) -> Program:
	body = bytes.fromhex(code)
	return build_program(
		Machine.EM_ARM,
		(
			("caller", 0x1000, len(body)),
			*((f"case_{address:x}", address, 4) for address in (0x2000, 0x3000, 0x5000, 0x6000)),
		),
		objects=(("table", 0x4000, _pointer(0x2000, 4) + _pointer(0x3000, 4)),),
		sections={0x1000: body},
		pointer_size=4,
		data_in_code=data_in_code,
	)


_TBB: Final = "02280cd8dfe800f00205080042f2000307e043f2000304e045f2000301e046f2000398477047"


@pytest.mark.parametrize(
	("code", "data_in_code"),
	[
		pytest.param(_TBB, (), id="tbb"),
		pytest.param(_TBB, ((0x1008, 0x100C),), id="tbb-with-a-marked-table"),
		pytest.param(
			"02280dd8dfe810f003000600090042f2000307e043f2000304e045f2000301e046f2000398477047",
			(),
			id="tbh",
		),
		pytest.param(
			"022812d801a252f820f000bf191000001f1000002510000042f2000307e043f2000304e045f2000301e0"
			"46f200039847704700bf",
			(),
			id="adr-and-ldr-pc",
		),
		pytest.param(
			"02280bd807a252f820f042f2000307e043f2000304e045f2000301e046f20003984770470b100000"
			"1110000017100000",
			(),
			id="a-table-after-the-cases",
		),
		pytest.param(
			"08a202280ad852f820f042f2000307e043f2000304e045f2000301e046f20003984770470b100000"
			"1110000017100000",
			(),
			id="adr-before-the-bound",
		),
	],
)
def test_thumb_bounded_dispatch_reaches_every_case(
	code: str, data_in_code: tuple[tuple[int, int], ...]
) -> None:
	(site,) = extract_call_sites(_thumb_dispatch_program(code, data_in_code))
	assert site.target == Known(
		values=frozenset(Address(address) for address in (0x2000, 0x3000, 0x5000, 0x6000))
	)


@pytest.mark.parametrize(
	"code",
	[
		pytest.param(
			"002901d002280cd8dfe800f00205080042f2000307e043f2000304e045f2000301e046f2000398477047",
			id="a-branch-into-the-dispatch-bypasses-the-bound",
		),
		pytest.param(
			"00290cd0dfe800f00205080042f2000307e043f2000304e045f2000301e046f2000398477047",
			id="no-bound",
		),
		pytest.param(
			"002906d0874442f2000304e043f2000301e046f2000398477047", id="add-pc-has-no-table"
		),
		pytest.param(
			"022809d8874442f2000307e043f2000304e045f2000301e046f2000398477047",
			id="add-pc-after-a-bound",
		),
		pytest.param(
			"dfe800f0000042f2000304e043f2000301e046f2000398477047", id="dispatch-before-any-guard"
		),
		pytest.param(
			"c82808d8dfe800f0010342f2000304e043f2000301e046f2000398477047",
			id="table-past-the-section",
		),
		pytest.param(
			"012808d8dfe800f07f7f42f2000304e043f2000301e046f2000398477047",
			id="targets-outside-the-function",
		),
		pytest.param(
			"012808d80a4652f820f042f2000304e043f2000301e046f2000398477047",
			id="ldr-pc-without-adr",
		),
		pytest.param(
			"02280cd8dfe800f00305080042f2000307e043f2000304e045f2000301e046f2000398477047",
			id="a-target-inside-an-instruction",
		),
	],
)
def test_thumb_unbounded_dispatch_reaches_every_block(code: str) -> None:
	sites = [
		site
		for site in extract_call_sites(_thumb_dispatch_program(code))
		if isinstance(site.target, Known)
	]
	assert sites == []


@pytest.mark.parametrize(
	("code", "candidates"),
	[
		pytest.param("00b5 42f20003 5df804fb 9847", set[int](), id="ldr-pc-from-sp-returns"),
		pytest.param("42f20003 f746 9847", set[int](), id="mov-pc-lr-returns"),
		pytest.param("42f20003 9f46", {0x2000}, id="mov-pc-is-a-site"),
		pytest.param("44f20003 d3f804f0", {0x3000}, id="ldr-pc-is-a-site"),
	],
)
def test_thumb_pc_write_is_a_return_or_a_site(code: str, candidates: set[int]) -> None:
	(site,) = extract_call_sites(_thumb_dispatch_program(code))
	assert site.target == _expected(candidates)


def _a32(words: str) -> bytes:
	return b"".join(bytes.fromhex(word)[::-1] for word in words.split())


@pytest.mark.parametrize(
	("arm_code", "sites"),
	[
		pytest.param(((0x1000, 0x1008),), [{0x2000}], id="a32-reads-pc-plus-eight"),
		pytest.param((), [], id="unmarked-decodes-as-thumb"),
	],
)
def test_a32_code_decodes_only_inside_its_mapping_span(
	arm_code: tuple[tuple[int, int], ...], sites: list[set[int]]
) -> None:
	body = _a32("e59f3000 e12fff33") + _pointer(0x2000, 4)
	program = build_program(
		Machine.EM_ARM,
		(("caller", 0x1000, len(body)), ("a", 0x2000, 4), ("b", 0x3000, 4)),
		sections={0x1000: body},
		pointer_size=4,
		data_in_code=((0x1008, 0x100C),),
		arm_code=arm_code,
	)
	assert [site.target for site in extract_call_sites(program)] == [
		_expected(expected) for expected in sites
	]


def test_a32_and_thumb_functions_each_decode_in_their_own_set() -> None:
	body = (
		_a32("e59f3000 e12fff33")
		+ _pointer(0x3001, 4)
		+ bytes.fromhex("004b 9847")
		+ _pointer(0x2000, 4)
	)
	program = build_program(
		Machine.EM_ARM,
		(
			("arm_caller", 0x1000, 0xC),
			("thumb_caller", 0x100D, 8),
			("a", 0x2000, 4),
			("b", 0x3001, 4),
		),
		sections={0x1000: body},
		pointer_size=4,
		data_in_code=((0x1008, 0x100C), (0x1010, 0x1014)),
		arm_code=((0x1000, 0x1008),),
	)
	assert sorted((site.site_address, site.target) for site in extract_call_sites(program)) == [
		(Address(0x1004), Known(values=frozenset({Address(0x3001)}))),
		(Address(0x100E), Known(values=frozenset({Address(0x2000)}))),
	]


def test_a32_pc_relative_load_with_an_index_is_unknown() -> None:
	body = _a32("e79f3102 e12fff33") + _pointer(0x2000, 4)
	program = build_program(
		Machine.EM_ARM,
		(("caller", 0x1000, len(body)), ("a", 0x2000, 4)),
		sections={0x1000: body},
		pointer_size=4,
		data_in_code=((0x1008, 0x100C),),
		arm_code=((0x1000, 0x1008),),
	)
	(site,) = extract_call_sites(program)
	assert site.target == Top()


@pytest.mark.parametrize(
	("restore", "candidates"),
	[
		pytest.param("e24dd000", {0x2000}, id="from-sp"),
		pytest.param("e28db004 e24bd004", {0x2000}, id="from-an-sp-copy"),
		pytest.param("e24bd000", set[int](), id="from-an-unknown-register"),
	],
)
def test_a32_sp_rebuilt_from_a_register_keeps_the_frame_only_when_it_is_an_sp_copy(
	restore: str, candidates: set[int]
) -> None:
	body = _a32(f"e3a03a02 e58d3000 {restore} e59d3000 e12fff33")
	program = build_program(
		Machine.EM_ARM,
		(("caller", 0x1000, len(body)), ("a", 0x2000, 4)),
		sections={0x1000: body},
		pointer_size=4,
		arm_code=((0x1000, 0x1000 + len(body)),),
	)
	(site,) = extract_call_sites(program)
	assert site.target == _expected(candidates)


@pytest.mark.parametrize(
	("caller", "body", "targets", "arm_code"),
	[
		pytest.param(
			0x1000, _a32("e28f0008 e24f0010"), (0xFFC, 0x1010), ((0xFFC, 0x1014),), id="a32-to-a32"
		),
		pytest.param(
			0x1000, _a32("e28f0008 e24f0010"), (0xFFD, 0x1011), ((0x1000, 0x1008),), id="a32-to-t32"
		),
		pytest.param(
			0x1001, bytes.fromhex("01a07047"), (0x1008,), ((0x1008, 0x100C),), id="t32-to-a32"
		),
		pytest.param(0x1001, bytes.fromhex("01a07047"), (0x1009,), (), id="t32-to-t32"),
	],
)
def test_a_pc_relative_address_takes_the_thumb_bit_of_its_target(
	caller: int, body: bytes, targets: tuple[int, ...], arm_code: tuple[tuple[int, int], ...]
) -> None:
	program = build_program(
		Machine.EM_ARM,
		(("caller", caller, len(body)), *((f"f_{target:x}", target, 4) for target in targets)),
		sections={0x1000: body},
		pointer_size=4,
		arm_code=arm_code,
	)
	assert address_taken(program) == frozenset(Address(target) for target in targets)


def test_instruction_runs_split_where_a32_code_ends() -> None:
	program = build_program(
		Machine.EM_ARM,
		(("mixed", 0x1000, 16),),
		sections={0x1000: bytes(16)},
		pointer_size=4,
		data_in_code=((0x1008, 0x100C),),
		arm_code=((0x1000, 0x1004),),
	)
	assert [
		(address, len(code), instruction_set_at(program, address))
		for address, code in instruction_runs(program, Address(0x1000), 16)
	] == [
		(0x1000, 4, InstructionSet.A32),
		(0x1004, 4, InstructionSet.T32),
		(0x100C, 4, InstructionSet.T32),
	]


def _a32_dispatch_program(
	code: str, data_in_code: tuple[tuple[int, int], ...], *, writable: bool = False
) -> Program:
	body = bytes.fromhex(code)
	return build_program(
		Machine.EM_ARM,
		(
			("caller", 0x1000, len(body)),
			*((f"case_{address:x}", address, 4) for address in (0x2000, 0x3000, 0x5000, 0x6000)),
		),
		sections={0x1000: body},
		writable=frozenset({0x1000}) if writable else frozenset(),
		pointer_size=4,
		data_in_code=data_in_code,
		arm_code=((0x1000, 0x1000 + len(body)),),
	)


@pytest.mark.parametrize(
	("code", "data_in_code"),
	[
		pytest.param(
			"2c309fe5020050e30600008a00f193e7001002e3040000ea001003e3020000ea001005e3000000ea"
			"001006e331ff2fe11eff2fe138100000101000001810000020100000",
			((0x1034, 0x1044),),
			id="word-table-via-a-literal",
		),
		pytest.param(
			"2c308fe2020050e30600008a00f193e7001002e3040000ea001003e3020000ea001005e3000000ea"
			"001006e331ff2fe11eff2fe1101000001810000020100000",
			((0x1034, 0x1040),),
			id="word-table-via-adr",
		),
		pytest.param(
			"383001e3003040e3020050e30600008a00f193e7001002e3040000ea001003e3020000ea001005e3"
			"000000ea001006e331ff2fe11eff2fe1141000001c10000024100000",
			((0x1038, 0x1044),),
			id="word-table-via-movw-movt",
		),
		pytest.param(
			"34309fe5020050e30800008a0000d3e700f18fe000f020e3001002e3040000ea001003e3020000ea"
			"001005e3000000ea001006e331ff2fe11eff2fe14010000000020400",
			((0x103C, 0x1044),),
			id="byte-offsets",
		),
		pytest.param(
			"3c309fe510402de9020050e30900008a000080e0b00093e100f18fe000f020e3001002e3040000ea"
			"001003e3020000ea001005e3000000ea001006e331ff2fe11080bde8481000000000020004000000",
			((0x1044, 0x1050),),
			id="halfword-offsets-after-a-push",
		),
		pytest.param(
			"38309fe5020050e30900008a8000a0e1b00093e100f18fe000f020e3001002e3040000ea001003e3"
			"020000ea001005e3000000ea001006e331ff2fe11eff2fe1441000000000020004000000",
			((0x1040, 0x104C),),
			id="halfword-offsets-scaled-by-lsl",
		),
	],
)
def test_a32_bounded_dispatch_reaches_every_case(
	code: str, data_in_code: tuple[tuple[int, int], ...]
) -> None:
	(site,) = extract_call_sites(_a32_dispatch_program(code, data_in_code))
	assert site.target == Known(
		values=frozenset(Address(address) for address in (0x2000, 0x3000, 0x5000, 0x6000))
	)


@pytest.mark.parametrize(
	("code", "data_in_code"),
	[
		pytest.param(
			"34309fe5020050e30600008a00f193e7001002e3040000ea001003e3020000ea001005e3000000ea"
			"001006e331ff2fe1012052e2f2ffff1a1eff2fe140100000101000001810000020100000",
			((0x103C, 0x104C),),
			id="a-branch-past-the-literal-load",
		),
		pytest.param(
			"38309fe5020050e30900008a000052e30000d31700f18fe000f020e3001002e3040000ea001003e3"
			"020000ea001005e3000000ea001006e331ff2fe11eff2fe14410000000020400",
			((0x1040, 0x1048),),
			id="a-predicated-index-load",
		),
		pytest.param(
			"30309fe5020050e30700008a003023e000f193e7001002e3040000ea001003e3020000ea001005e3"
			"000000ea001006e331ff2fe11eff2fe13c100000141000001c10000024100000",
			((0x1038, 0x1048),),
			id="a-clobbered-base",
		),
		pytest.param(
			"34309fe5020050e30800008a0000d3e700f18f9000f020e3001002e3040000ea001003e3020000ea"
			"001005e3000000ea001006e331ff2fe11eff2fe14010000000020400",
			((0x103C, 0x1044),),
			id="a-predicated-dispatch",
		),
		pytest.param(
			"2c309fe5020050e30600008a00f113e7001002e3040000ea001003e3020000ea001005e3000000ea"
			"001006e331ff2fe11eff2fe138100000101000001810000020100000",
			((0x1034, 0x1044),),
			id="a-subtracted-index",
		),
		pytest.param(
			"34309fe5020050e30800008a0000d3e720f18fe000f020e3001002e3040000ea001003e3020000ea"
			"001005e3000000ea001006e331ff2fe11eff2fe14010000000020400",
			((0x103C, 0x1044),),
			id="a-shift-other-than-lsl",
		),
		pytest.param(
			"30309fe5020050e30700008a00009de500f193e7001002e3040000ea001003e3020000ea001005e3"
			"000000ea001006e331ff2fe11eff2fe13c100000141000001c10000024100000",
			((0x1038, 0x1048),),
			id="an-index-from-memory",
		),
		pytest.param(
			"30309fe5020050e30700008a1002a0e100f093e7001002e3040000ea001003e3020000ea001005e3"
			"000000ea001006e331ff2fe11eff2fe13c100000141000001c10000024100000",
			((0x1038, 0x1048),),
			id="a-register-shifted-index",
		),
		pytest.param(
			"003040e3020050e30600008a00f193e7001002e3040000ea001003e3020000ea001005e3000000ea"
			"001006e331ff2fe11eff2fe1101000001810000020100000",
			((0x1034, 0x1040),),
			id="movt-of-an-unknown-register",
		),
	],
)
def test_a32_unbounded_dispatch_reaches_every_block(
	code: str, data_in_code: tuple[tuple[int, int], ...]
) -> None:
	sites = extract_call_sites(_a32_dispatch_program(code, data_in_code))
	assert [site for site in sites if isinstance(site.target, Known)] == []


@pytest.mark.parametrize(
	("writable", "candidates"),
	[
		pytest.param(False, {0x2000, 0x3000, 0x5000, 0x6000}, id="read-only"),
		pytest.param(True, set[int](), id="writable"),
	],
)
def test_a32_dispatch_reads_its_table_only_from_read_only_memory(
	writable: bool, candidates: set[int]
) -> None:
	program = _a32_dispatch_program(
		"2c309fe5020050e30600008a00f193e7001002e3040000ea001003e3020000ea001005e3000000ea"
		"001006e331ff2fe11eff2fe138100000101000001810000020100000",
		((0x1034, 0x1044),),
		writable=writable,
	)
	(site,) = extract_call_sites(program)
	assert site.target == _expected(candidates)


@pytest.mark.parametrize(
	("compare", "candidates"),
	[
		pytest.param("e3500fff", {0x2000, 0x6000}, id="1021-cases"),
		pytest.param("e3500b01", set[int](), id="1025-cases"),
	],
)
def test_a32_dispatch_past_the_case_limit_is_unbounded(compare: str, candidates: set[int]) -> None:
	code = _a32(
		f"e59f302c {compare} 8a000006 e793f100 e3021000 ea000004 e3031000 ea000002 e3051000"
		" ea000000 e3061000 e12fff31 e12fff1e"
	)
	program = _a32_dispatch_program(
		(code + _pointer(0x1038, 4) + _pointer(0x1010, 4) * 1025).hex(),
		((0x1034, 0x1038 + 4 * 1025),),
	)
	(site,) = extract_call_sites(program)
	assert site.target == _expected(candidates)


def _expected(candidates: Collection[int]) -> ValueSet:
	return (
		Known(values=frozenset(Address(address) for address in candidates))
		if frozenset(candidates)
		else Top()
	)
