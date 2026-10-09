# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Arm branch conditions, decided against signed and unsigned integer comparisons."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, assert_never

import pytest
from capstone import arm_const
from salix import Struct

from dynamic_call_tree_resolution import Address
from dynamic_call_tree_resolution.vsa.conditions import (
	Compared,
	Comparison,
	Immediate,
	InRegister,
	branch_taken,
)

if TYPE_CHECKING:
	from collections.abc import Callable, Mapping

_WORD: Final = 1 << 32

_PAIRS: Final = (
	(5, 5),
	(5, 6),
	(6, 5),
	(0, 0),
	(0xFFFF_FFFF, 1),
	(1, 0xFFFF_FFFF),
	(0x7FFF_FFFF, 0xFFFF_FFFF),
	(0x8000_0000, 1),
	(0x8000_0000, 0x8000_0000),
)


def _signed(value: int) -> int:
	return value - _WORD if value >> 31 else value


class _Outcome(Struct):
	exact: int
	carry: bool
	wrapped: int


def _outcome(comparison: Comparison, left: int, right: int) -> _Outcome:
	match comparison:
		case Comparison.SUBTRACT:
			return _Outcome(
				exact=_signed(left) - _signed(right),
				carry=left >= right,
				wrapped=(left - right) % _WORD,
			)
		case Comparison.ADD:
			return _Outcome(
				exact=_signed(left) + _signed(right),
				carry=left + right >= _WORD,
				wrapped=(left + right) % _WORD,
			)
		case _ as unreachable:
			assert_never(unreachable)


_ORACLE: Final[Mapping[int, Callable[[_Outcome], bool]]] = {
	arm_const.ARM_CC_EQ: lambda outcome: outcome.wrapped == 0,
	arm_const.ARM_CC_NE: lambda outcome: outcome.wrapped != 0,
	arm_const.ARM_CC_HS: lambda outcome: outcome.carry,
	arm_const.ARM_CC_LO: lambda outcome: not outcome.carry,
	arm_const.ARM_CC_HI: lambda outcome: outcome.carry and outcome.wrapped != 0,
	arm_const.ARM_CC_LS: lambda outcome: not outcome.carry or outcome.wrapped == 0,
	arm_const.ARM_CC_MI: lambda outcome: outcome.wrapped >= _WORD // 2,
	arm_const.ARM_CC_PL: lambda outcome: outcome.wrapped < _WORD // 2,
	arm_const.ARM_CC_VS: lambda outcome: not -_WORD // 2 <= outcome.exact < _WORD // 2,
	arm_const.ARM_CC_VC: lambda outcome: -_WORD // 2 <= outcome.exact < _WORD // 2,
	arm_const.ARM_CC_GE: lambda outcome: outcome.exact >= 0,
	arm_const.ARM_CC_LT: lambda outcome: outcome.exact < 0,
	arm_const.ARM_CC_GT: lambda outcome: outcome.exact > 0,
	arm_const.ARM_CC_LE: lambda outcome: outcome.exact <= 0,
}


@pytest.mark.parametrize(
	("comparison", "condition", "left", "right"),
	[
		(comparison, condition, left, right)
		for comparison in Comparison
		for condition in _ORACLE
		for left, right in _PAIRS
	],
)
def test_a_known_compare_decides_its_branch_as_integer_comparison_does(
	comparison: Comparison, condition: int, left: int, right: int
) -> None:
	assert branch_taken(
		Compared(
			comparison=comparison,
			left=arm_const.ARM_REG_R0,
			right=Immediate(value=right),
			condition=condition,
		),
		{arm_const.ARM_REG_R0: frozenset({Address(left)})},
	) == _ORACLE[condition](_outcome(comparison, left, right))


@pytest.mark.parametrize(
	("left", "right", "taken"),
	[
		pytest.param(frozenset({5, 7}), frozenset({6}), True, id="every value agrees"),
		pytest.param(frozenset({5, 6}), frozenset({6}), None, id="values disagree"),
		pytest.param(None, frozenset({6}), None, id="an unknown left operand"),
		pytest.param(frozenset({6}), None, None, id="an unknown right operand"),
		pytest.param(frozenset[int](), frozenset({6}), None, id="no value"),
	],
)
def test_a_branch_is_decided_only_when_every_operand_value_agrees(
	left: frozenset[int] | None, right: frozenset[int] | None, taken: bool | None
) -> None:
	assert (
		branch_taken(
			Compared(
				comparison=Comparison.SUBTRACT,
				left=arm_const.ARM_REG_R0,
				right=InRegister(register=arm_const.ARM_REG_R1),
				condition=arm_const.ARM_CC_NE,
			),
			{
				register: frozenset(map(Address, values))
				for register, values in (
					(arm_const.ARM_REG_R0, left),
					(arm_const.ARM_REG_R1, right),
				)
				if values is not None
			},
		)
		is taken
	)
