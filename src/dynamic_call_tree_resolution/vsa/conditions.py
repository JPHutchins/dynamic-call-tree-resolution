# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Conditional branches whose direction the tracked register values decide."""

from __future__ import annotations

from enum import IntEnum
from itertools import product
from typing import TYPE_CHECKING, Final, assert_never, cast

from capstone import arm_const
from salix import Struct

from dynamic_call_tree_resolution.vsa.lattice import Known, Top, lookup

if TYPE_CHECKING:
	from collections.abc import Callable, Mapping

	from capstone import ArmCsOperand, CsInsn

	from dynamic_call_tree_resolution.model import Address

_WORD_MASK: Final = 0xFFFF_FFFF
_SIGN_SHIFT: Final = 31


class Comparison(IntEnum):
	"""A flag-setting compare, by its capstone instruction id."""

	SUBTRACT = arm_const.ARM_INS_CMP
	ADD = arm_const.ARM_INS_CMN


class InRegister(Struct):
	"""A compare operand a register holds."""

	register: int


class Immediate(Struct):
	"""A compare operand the instruction encodes."""

	value: int


class Compared(Struct):
	"""A conditional branch on the flags a compare earlier in its own block sets."""

	comparison: Comparison
	left: int
	right: InRegister | Immediate
	condition: int
	"""The branch's capstone ``ARM_CC_*`` code."""


class ZeroTested(Struct):
	"""A ``cbz`` or ``cbnz``."""

	register: int
	taken_when_zero: bool


type BranchTest = Compared | ZeroTested


class Conditional(Struct):
	"""A block-ending conditional branch or predicated exit that the values may decide."""

	test: BranchTest
	taken: tuple[Address, ...]
	"""Empty when the branch leaves the function."""
	fallthrough: tuple[Address, ...]


class _Flags(Struct):
	negative: bool
	zero: bool
	carry: bool
	overflow: bool


_CONDITIONS: Final[Mapping[int, Callable[[_Flags], bool]]] = {
	arm_const.ARM_CC_EQ: lambda flags: flags.zero,
	arm_const.ARM_CC_NE: lambda flags: not flags.zero,
	arm_const.ARM_CC_HS: lambda flags: flags.carry,
	arm_const.ARM_CC_LO: lambda flags: not flags.carry,
	arm_const.ARM_CC_MI: lambda flags: flags.negative,
	arm_const.ARM_CC_PL: lambda flags: not flags.negative,
	arm_const.ARM_CC_VS: lambda flags: flags.overflow,
	arm_const.ARM_CC_VC: lambda flags: not flags.overflow,
	arm_const.ARM_CC_HI: lambda flags: flags.carry and not flags.zero,
	arm_const.ARM_CC_LS: lambda flags: not flags.carry or flags.zero,
	arm_const.ARM_CC_GE: lambda flags: flags.negative == flags.overflow,
	arm_const.ARM_CC_LT: lambda flags: flags.negative != flags.overflow,
	arm_const.ARM_CC_GT: lambda flags: not flags.zero and flags.negative == flags.overflow,
	arm_const.ARM_CC_LE: lambda flags: flags.zero or flags.negative != flags.overflow,
}


def arm_branch_test(instructions: tuple[CsInsn, ...]) -> BranchTest | None:
	"""What a block's final conditional transfer tests, when its own block shows it."""
	branch = instructions[-1]
	match branch.id:
		case arm_const.ARM_INS_CBZ | arm_const.ARM_INS_CBNZ as zero_test:
			return ZeroTested(
				register=branch.operands[0].reg, taken_when_zero=zero_test == arm_const.ARM_INS_CBZ
			)
		case _:
			return _compared(instructions[:-1], branch.cc)


def _sets_flags(instruction: CsInsn) -> bool:
	return instruction.update_flags or instruction.id == arm_const.ARM_INS_MSR


def _compared(preceding: tuple[CsInsn, ...], condition: int) -> Compared | None:
	setter = next(
		(index for index in range(len(preceding) - 1, -1, -1) if _sets_flags(preceding[index])),
		None,
	)
	return (
		None
		if setter is None
		else _compare(
			preceding[setter],
			frozenset(
				register
				for instruction in preceding[setter + 1 :]
				for register in instruction.regs_access()[1]
			),
			condition,
		)
	)


def _compare(setter: CsInsn, written_after: frozenset[int], condition: int) -> Compared | None:
	operands = tuple(cast("ArmCsOperand", operand) for operand in setter.operands)
	return (
		Compared(
			comparison=Comparison(setter.id),
			left=operands[0].reg,
			right=InRegister(register=operands[1].reg)
			if operands[1].type == arm_const.ARM_OP_REG
			else Immediate(value=operands[1].imm),
			condition=condition,
		)
		if setter.id in Comparison
		and condition in _CONDITIONS
		and setter.cc == arm_const.ARM_CC_AL
		and all(operand.shift.type == arm_const.ARM_SFT_INVALID for operand in operands)
		and operands[1].type in (arm_const.ARM_OP_REG, arm_const.ARM_OP_IMM)
		and written_after.isdisjoint(setter.regs_access()[0])
		else None
	)


def edge_values(
	test: BranchTest, registers: Mapping[int, frozenset[Address]]
) -> Mapping[bool, Mapping[int, frozenset[Address]]]:
	"""The tested registers' values on each outcome's edge; an outcome no value takes is absent.

	A test reading a register the analysis knows no value of narrows nothing: both outcomes keep
	every register whole.
	"""
	match _assignments(test, registers):
		case None:
			return {True: {}, False: {}}
		case assignments:
			outcomes = tuple((_taken(test, assignment), assignment) for assignment in assignments)
			return {
				outcome: {
					register: frozenset(
						assignment[register] for taken, assignment in outcomes if taken == outcome
					)
					for register in _registers(test)
				}
				for outcome in frozenset(taken for taken, _ in outcomes)
			}


def _assignments(
	test: BranchTest, registers: Mapping[int, frozenset[Address]]
) -> tuple[Mapping[int, Address], ...] | None:
	"""Every combination of values the tested registers hold; None when one holds none known."""
	known = tuple(_values(registers, register) for register in _registers(test))
	return (
		tuple(
			dict(zip(_registers(test), combination, strict=True)) for combination in product(*known)
		)
		if all(known)
		else None
	)


def _registers(test: BranchTest) -> tuple[int, ...]:
	match test:
		case ZeroTested(register=register):
			return (register,)
		case Compared(left=left, right=InRegister(register=right)):
			return (left,) if left == right else (left, right)
		case Compared(left=left):
			return (left,)
		case _ as unreachable:
			assert_never(unreachable)


def _taken(test: BranchTest, values: Mapping[int, Address]) -> bool:
	match test:
		case ZeroTested(register=register, taken_when_zero=taken_when_zero):
			return (values[register] & _WORD_MASK == 0) == taken_when_zero
		case Compared(comparison=comparison, left=left, right=right, condition=condition):
			return _CONDITIONS[condition](
				_flags(comparison, values[left] & _WORD_MASK, _operand(values, right) & _WORD_MASK)
			)
		case _ as unreachable:
			assert_never(unreachable)


def _operand(values: Mapping[int, Address], operand: InRegister | Immediate) -> int:
	match operand:
		case InRegister(register=register):
			return values[register]
		case Immediate(value=value):
			return value
		case _ as unreachable:
			assert_never(unreachable)


def _values(registers: Mapping[int, frozenset[Address]], register: int) -> frozenset[Address]:
	match lookup(registers, register):
		case Top():
			return frozenset()
		case Known(values=values):
			return values
		case _ as unreachable:
			assert_never(unreachable)


def _flags(comparison: Comparison, left: int, right: int) -> _Flags:
	match comparison:
		case Comparison.SUBTRACT:
			return _added(left, ~right & _WORD_MASK, 1)
		case Comparison.ADD:
			return _added(left, right, 0)
		case _ as unreachable:
			assert_never(unreachable)


def _added(left: int, right: int, carry_in: int) -> _Flags:
	"""The flags of Arm's AddWithCarry.

	>>> _added(0x412C, ~0x412C & _WORD_MASK, 1)
	_Flags(negative=False, zero=True, carry=True, overflow=False)
	>>> _added(0x7FFF_FFFF, 1, 0)
	_Flags(negative=True, zero=False, carry=False, overflow=True)
	"""
	return _sum_flags(left, right, left + right + carry_in)


def _sum_flags(left: int, right: int, total: int) -> _Flags:
	return _Flags(
		negative=bool(total >> _SIGN_SHIFT & 1),
		zero=total & _WORD_MASK == 0,
		carry=total > _WORD_MASK,
		overflow=bool((~(left ^ right) & (left ^ total)) >> _SIGN_SHIFT & 1),
	)
