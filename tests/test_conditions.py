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
	ZeroTested,
	edge_values,
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
	assert set(
		edge_values(
			Compared(
				comparison=comparison,
				left=arm_const.ARM_REG_R0,
				right=Immediate(value=right),
				condition=condition,
			),
			{arm_const.ARM_REG_R0: frozenset({Address(left)})},
		)
	) == {_ORACLE[condition](_outcome(comparison, left, right))}


@pytest.mark.parametrize(
	("left", "right", "outcomes"),
	[
		pytest.param(frozenset({5, 7}), frozenset({6}), {True}, id="every value agrees"),
		pytest.param(frozenset({5, 6}), frozenset({6}), {True, False}, id="values disagree"),
		pytest.param(None, frozenset({6}), {True, False}, id="an unknown left operand"),
		pytest.param(frozenset({6}), None, {True, False}, id="an unknown right operand"),
		pytest.param(frozenset[int](), frozenset({6}), {True, False}, id="no value"),
	],
)
def test_a_branch_is_decided_only_when_every_operand_value_agrees(
	left: frozenset[int] | None, right: frozenset[int] | None, outcomes: set[bool]
) -> None:
	assert (
		set(
			edge_values(
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
		)
		== outcomes
	)


@pytest.mark.parametrize(
	("test", "registers", "edges"),
	[
		pytest.param(
			Compared(
				comparison=Comparison.SUBTRACT,
				left=arm_const.ARM_REG_R0,
				right=Immediate(value=6),
				condition=arm_const.ARM_CC_LO,
			),
			{arm_const.ARM_REG_R0: {5, 6, 7}},
			{True: {arm_const.ARM_REG_R0: {5}}, False: {arm_const.ARM_REG_R0: {6, 7}}},
			id="against an immediate",
		),
		pytest.param(
			Compared(
				comparison=Comparison.SUBTRACT,
				left=arm_const.ARM_REG_R0,
				right=InRegister(register=arm_const.ARM_REG_R1),
				condition=arm_const.ARM_CC_LO,
			),
			{arm_const.ARM_REG_R0: {5, 6, 7}, arm_const.ARM_REG_R1: {6, 7}},
			{
				True: {arm_const.ARM_REG_R0: {5, 6}, arm_const.ARM_REG_R1: {6, 7}},
				False: {arm_const.ARM_REG_R0: {6, 7}, arm_const.ARM_REG_R1: {6, 7}},
			},
			id="both registers",
		),
		pytest.param(
			Compared(
				comparison=Comparison.SUBTRACT,
				left=arm_const.ARM_REG_R0,
				right=InRegister(register=arm_const.ARM_REG_R1),
				condition=arm_const.ARM_CC_HS,
			),
			{arm_const.ARM_REG_R0: {0x4130, 0x4160, 0x4190}, arm_const.ARM_REG_R1: {0x4190}},
			{
				True: {arm_const.ARM_REG_R0: {0x4190}, arm_const.ARM_REG_R1: {0x4190}},
				False: {arm_const.ARM_REG_R0: {0x4130, 0x4160}, arm_const.ARM_REG_R1: {0x4190}},
			},
			id="a pointer walking to its end",
		),
		pytest.param(
			Compared(
				comparison=Comparison.SUBTRACT,
				left=arm_const.ARM_REG_R0,
				right=InRegister(register=arm_const.ARM_REG_R0),
				condition=arm_const.ARM_CC_EQ,
			),
			{arm_const.ARM_REG_R0: {3, 4}},
			{True: {arm_const.ARM_REG_R0: {3, 4}}},
			id="a register against itself",
		),
		pytest.param(
			ZeroTested(register=arm_const.ARM_REG_R0, taken_when_zero=True),
			{arm_const.ARM_REG_R0: {0, 4}},
			{True: {arm_const.ARM_REG_R0: {0}}, False: {arm_const.ARM_REG_R0: {4}}},
			id="cbz",
		),
		pytest.param(
			Compared(
				comparison=Comparison.SUBTRACT,
				left=arm_const.ARM_REG_R0,
				right=InRegister(register=arm_const.ARM_REG_R1),
				condition=arm_const.ARM_CC_LO,
			),
			{arm_const.ARM_REG_R0: {5, 6}},
			{True: {}, False: {}},
			id="an unknown operand narrows nothing",
		),
	],
)
def test_each_edge_keeps_the_values_that_take_it(
	test: Compared | ZeroTested,
	registers: Mapping[int, set[int]],
	edges: Mapping[bool, Mapping[int, set[int]]],
) -> None:
	assert edge_values(
		test,
		{register: frozenset(map(Address, values)) for register, values in registers.items()},
	) == {
		outcome: {register: frozenset(values) for register, values in kept.items()}
		for outcome, kept in edges.items()
	}
