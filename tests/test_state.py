# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Where a frame address escapes, as the analysis tracks it."""

from typing import Final

import pytest
from capstone import arm_const
from salix import replace

from dynamic_call_tree_resolution.model import Address, Machine
from dynamic_call_tree_resolution.vsa.cfg import control_flow_graphs
from dynamic_call_tree_resolution.vsa.interpret import block_states
from dynamic_call_tree_resolution.vsa.memory import context_for
from dynamic_call_tree_resolution.vsa.state import (
	ESCAPED_ANYWHERE,
	NO_WRITES,
	UNESCAPED,
	EscapedFrom,
	FrameEscape,
	escape_after_join,
	escaping,
	frame_after_escape,
	join_escapes,
	top_seed,
)
from tests.programs import build_program


@pytest.mark.parametrize(
	("current", "incoming", "joined"),
	[
		pytest.param(UNESCAPED, UNESCAPED, UNESCAPED, id="neither"),
		pytest.param(UNESCAPED, EscapedFrom(-8), EscapedFrom(-8), id="only the incoming"),
		pytest.param(EscapedFrom(-8), UNESCAPED, EscapedFrom(-8), id="only the current"),
		pytest.param(EscapedFrom(-8), EscapedFrom(-16), EscapedFrom(-16), id="the lower of two"),
		pytest.param(EscapedFrom(-8), ESCAPED_ANYWHERE, ESCAPED_ANYWHERE, id="anywhere absorbs"),
		pytest.param(ESCAPED_ANYWHERE, EscapedFrom(-8), ESCAPED_ANYWHERE, id="either way"),
	],
)
def test_two_paths_escape_from_the_lower_of_their_addresses(
	current: FrameEscape, incoming: FrameEscape, joined: FrameEscape
) -> None:
	assert join_escapes(current, incoming) == joined


@pytest.mark.parametrize(
	("escaped", "registers", "after"),
	[
		pytest.param(UNESCAPED, (arm_const.ARM_REG_R1,), UNESCAPED, id="no frame address"),
		pytest.param(UNESCAPED, (arm_const.ARM_REG_R2,), UNESCAPED, id="no offset it can hold"),
		pytest.param(UNESCAPED, (arm_const.ARM_REG_R0,), EscapedFrom(-24), id="the lowest one"),
		pytest.param(EscapedFrom(-64), (arm_const.ARM_REG_R0,), EscapedFrom(-64), id="kept lower"),
		pytest.param(ESCAPED_ANYWHERE, (arm_const.ARM_REG_R0,), ESCAPED_ANYWHERE, id="anywhere"),
	],
)
def test_a_frame_address_escapes_from_the_lowest_offset_its_register_holds(
	escaped: FrameEscape, registers: tuple[int, ...], after: FrameEscape
) -> None:
	state = replace(
		top_seed(Machine.EM_ARM),
		sp_offsets={
			arm_const.ARM_REG_SP: frozenset({0}),
			arm_const.ARM_REG_R0: frozenset({-8, -24}),
			arm_const.ARM_REG_R2: frozenset(),
		},
		escaped=escaped,
	)
	assert escaping(state, registers, lowest_tracked=True).escaped == after


def test_without_tracking_the_lowest_address_any_frame_escape_is_anywhere() -> None:
	state = replace(
		top_seed(Machine.EM_ARM),
		sp_offsets={arm_const.ARM_REG_SP: frozenset({0}), arm_const.ARM_REG_R0: frozenset({-8})},
	)
	assert (
		escaping(state, (arm_const.ARM_REG_R0,), lowest_tracked=False).escaped,
		escaping(state, (arm_const.ARM_REG_R1,), lowest_tracked=False).escaped,
	) == (ESCAPED_ANYWHERE, UNESCAPED)


@pytest.mark.parametrize(
	("tracked", "escaped"),
	[
		pytest.param(False, ESCAPED_ANYWHERE, id="the default"),
		pytest.param(True, EscapedFrom(-8), id="tracked"),
	],
)
def test_a_call_passing_a_frame_address_records_where_it_escaped_only_when_tracked(
	tracked: bool, escaped: FrameEscape
) -> None:
	"""``sub sp, #16; add r0, sp, #8; bl callee; add sp, #16; bx lr``."""
	code = bytes.fromhex("84b0 02a8 00f002f8 04b0 7047 7047")
	program = build_program(
		Machine.EM_ARM,
		(("caller", 0x1000, 12), ("callee", 0x100C, 2)),
		sections={0x1000: code},
		pointer_size=4,
	)
	assert (
		block_states(
			context_for(program, NO_WRITES, upward_frame_writes=tracked),
			control_flow_graphs(program)[Address(0x1000)].blocks,
			top_seed(Machine.EM_ARM),
		)
		.in_states[Address(0x1008)]
		.escaped
		== escaped
	)


@pytest.mark.parametrize(
	("escaped", "kept"),
	[
		pytest.param(UNESCAPED, (-20, -16, -12, -11, -8), id="unescaped"),
		pytest.param(EscapedFrom(-12), (-20, -16), id="the words wholly below"),
		pytest.param(ESCAPED_ANYWHERE, (), id="anywhere"),
	],
)
def test_after_an_escape_only_the_words_below_the_lowest_escaped_address_stay_known(
	escaped: FrameEscape, kept: tuple[int, ...]
) -> None:
	state = replace(
		top_seed(Machine.EM_ARM),
		stack={offset: frozenset({Address(5)}) for offset in (-20, -16, -12, -11, -8)},
		escaped=escaped,
	)
	assert tuple(frame_after_escape(state, 4)) == kept


_SPILLED: Final = "00b5 84b0 0521 0091 0191 0291 02a8"
"""``push {lr}; sub sp, #16; movs r1, #5``, stored at entry - 20, - 16 and - 12, then
``add r0, sp, #8``."""


@pytest.mark.parametrize(
	"escape",
	[
		pytest.param("00f002f8 04b0 00bd 7047", id="passed to a call"),
		pytest.param("1060 ffe7 04b0 00bd 7047", id="stored through an unknown pointer"),
	],
)
@pytest.mark.parametrize(
	("upward", "stack"),
	[
		pytest.param(False, {}, id="the default"),
		pytest.param(
			True,
			{-20: frozenset({Address(5)}), -16: frozenset({Address(5)})},
			id="upward frame writes",
		),
	],
)
def test_a_frame_address_escaping_keeps_the_slots_below_it_only_with_upward_frame_writes(
	escape: str, upward: bool, stack: dict[int, frozenset[Address]]
) -> None:
	code = bytes.fromhex(f"{_SPILLED} {escape}")
	program = build_program(
		Machine.EM_ARM,
		(("caller", 0x1000, 22), ("callee", 0x1016, 2)),
		sections={0x1000: code},
		pointer_size=4,
	)
	assert (
		block_states(
			context_for(program, NO_WRITES, upward_frame_writes=upward),
			control_flow_graphs(program)[Address(0x1000)].blocks,
			top_seed(Machine.EM_ARM),
		)
		.in_states[Address(0x1012)]
		.stack
		== stack
	)


@pytest.mark.parametrize(
	("instruction", "upward", "escaped"),
	[
		pytest.param("4140", False, ESCAPED_ANYWHERE, id="eors r1, r0"),
		pytest.param("4140", True, EscapedFrom(-8), id="eors r1, r0, upward"),
		pytest.param("1118", False, ESCAPED_ANYWHERE, id="adds r1, r2, r0"),
		pytest.param("1118", True, EscapedFrom(-8), id="adds r1, r2, r0, upward"),
		pytest.param("111a", False, ESCAPED_ANYWHERE, id="subs r1, r2, r0"),
		pytest.param("111a", True, UNESCAPED, id="subs r1, r2, r0, upward: a difference"),
		pytest.param("811a", False, ESCAPED_ANYWHERE, id="subs r1, r0, r2"),
		pytest.param("811a", True, ESCAPED_ANYWHERE, id="subs r1, r0, r2, upward: below r0"),
	],
)
def test_a_value_computed_from_a_frame_address_escapes_from_it_with_upward_frame_writes(
	instruction: str, upward: bool, escaped: FrameEscape
) -> None:
	"""``sub sp, #16; add r0, sp, #8``, the instruction, then a branch to the next block."""
	code = bytes.fromhex(f"84b0 02a8 {instruction} ffe7 04b0 7047")
	program = build_program(
		Machine.EM_ARM, (("caller", 0x1000, 12),), sections={0x1000: code}, pointer_size=4
	)
	assert (
		block_states(
			context_for(program, NO_WRITES, upward_frame_writes=upward),
			control_flow_graphs(program)[Address(0x1000)].blocks,
			top_seed(Machine.EM_ARM),
		)
		.in_states[Address(0x1008)]
		.escaped
		== escaped
	)


@pytest.mark.parametrize(
	("incoming", "lowest_tracked", "escaped"),
	[
		pytest.param({13: frozenset({-16})}, False, ESCAPED_ANYWHERE, id="a copy lost"),
		pytest.param({13: frozenset({-16})}, True, EscapedFrom(-8), id="a copy lost, tracked"),
		pytest.param(
			{13: frozenset({-16}), 0: frozenset({-24})}, True, UNESCAPED, id="no copy lost"
		),
	],
)
def test_a_frame_copy_a_join_drops_escapes_from_its_offsets_when_tracked(
	incoming: dict[int, frozenset[int]], lowest_tracked: bool, escaped: FrameEscape
) -> None:
	current = {13: frozenset({-16}), 0: frozenset({-8})}
	joined = {
		register: current[register] | offsets
		for register, offsets in incoming.items()
		if register in current
	}
	assert (
		escape_after_join(UNESCAPED, current, incoming, joined, lowest_tracked=lowest_tracked)
		== escaped
	)
