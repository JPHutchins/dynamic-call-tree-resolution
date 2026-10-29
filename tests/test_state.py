# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Where a frame address escapes, as the analysis tracks it."""

import pytest
from capstone import arm_const
from salix import replace

from dynamic_call_tree_resolution.model import Machine
from dynamic_call_tree_resolution.vsa.state import (
	ESCAPED_ANYWHERE,
	UNESCAPED,
	EscapedFrom,
	FrameEscape,
	escaping,
	join_escapes,
	top_seed,
)


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
