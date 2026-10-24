# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Where a block's branch test narrows, and which moved copies it narrows with it (#185)."""

from __future__ import annotations

import pytest
from capstone import CS_ARCH_ARM, CS_MODE_THUMB, Cs, arm_const

from dynamic_call_tree_resolution.vsa.cfg.narrowing import Narrowing, RegisterCopy, narrowing
from dynamic_call_tree_resolution.vsa.conditions import arm_branch_test


def _narrowing_of(code: str) -> Narrowing | None:
	disassembler = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
	disassembler.detail = True
	instructions = tuple(disassembler.disasm(bytes.fromhex(code), 0x1000))
	test = arm_branch_test(instructions)
	return None if test is None else narrowing(instructions, test)


@pytest.mark.parametrize(
	("code", "expected"),
	[
		pytest.param(
			"2646ae4204f10404fad3",
			Narrowing(
				at=1,
				copies=(RegisterCopy(register=arm_const.ARM_REG_R6, source=arm_const.ARM_REG_R4),),
			),
			id="mov r6, r4; cmp r6, r5; add.w r4, #4; blo",
		),
		pytest.param(
			"264604f10404ae42fad3",
			Narrowing(
				at=1,
				copies=(RegisterCopy(register=arm_const.ARM_REG_R6, source=arm_const.ARM_REG_R4),),
			),
			id="mov r6, r4; add.w r4, #4; cmp r6, r5; blo",
		),
		pytest.param("ac4204f10104fbd1", Narrowing(at=0), id="cmp r4, r5; add.w r4, #1; bne"),
		pytest.param("012100b1", Narrowing(at=1), id="movs r1, #1; cbz r0"),
	],
)
def test_a_test_narrows_where_its_registers_last_changed(code: str, expected: Narrowing) -> None:
	assert _narrowing_of(code) == expected


@pytest.mark.parametrize(
	("code", "at"),
	[
		pytest.param("002808bf2646ae42fad3", 3, id="a predicated move"),
		pytest.param("2046fff7fdffa842fad3", 2, id="a call clobbering the copy"),
		pytest.param("26463c687d68ae42fad3", 3, id="a source loaded over before the test reads"),
		pytest.param("2e46ae42fcd3", 1, id="a source the test reads too"),
		pytest.param("6e46ae42fcd3", 1, id="the stack pointer"),
	],
)
def test_a_move_whose_source_may_differ_where_the_test_narrows_is_no_copy(
	code: str, at: int
) -> None:
	assert _narrowing_of(code) == Narrowing(at=at)
