# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Where in its block a branch test's registers last changed, and what they were copied from."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, assert_never, cast

from capstone import arm_const
from salix import Struct

from dynamic_call_tree_resolution.vsa.abi import (
	ARM_CALLER_SAVED,
	ARM_CALLS,
	ARM_RETURNING_TRAPS,
	arm_predicated,
)
from dynamic_call_tree_resolution.vsa.conditions import Compared, ZeroTested, tested_registers

if TYPE_CHECKING:
	from capstone import ArmCsOperand, CsInsn

	from dynamic_call_tree_resolution.vsa.conditions import BranchTest

_CLOBBERING: Final = ARM_CALLS | ARM_RETURNING_TRAPS


class RegisterCopy(Struct):
	"""A tested register a move set, and the register it holds a copy of from there on."""

	register: int
	source: int


class Narrowing(Struct):
	"""Where a branch's outcome narrows the registers its test reads."""

	at: int
	"""The index of the first instruction from which they hold the values the test reads."""
	copies: tuple[RegisterCopy, ...] = ()


def narrowing(instructions: tuple[CsInsn, ...], test: BranchTest) -> Narrowing:
	match test:
		case ZeroTested():
			return Narrowing(at=len(instructions) - 1)
		case Compared(setter=setter):
			return _narrowing(instructions[:setter], tested_registers(test))
		case _ as unreachable:
			assert_never(unreachable)


def _narrowing(preceding: tuple[CsInsn, ...], tested: tuple[int, ...]) -> Narrowing:
	writers = {register: _last_writer(preceding, register) for register in tested}
	at = max((index + 1 for index in writers.values() if index is not None), default=len(preceding))
	return Narrowing(
		at=at,
		copies=tuple(
			RegisterCopy(register=register, source=source)
			for register, index in writers.items()
			if index is not None
			for source in (_move_source(preceding[index]),)
			if source is not None
			and source not in tested
			and _last_writer(preceding[index + 1 : at], source) is None
		),
	)


def _last_writer(instructions: tuple[CsInsn, ...], register: int) -> int | None:
	return next(
		(
			index
			for index in range(len(instructions) - 1, -1, -1)
			if register in _written(instructions[index])
		),
		None,
	)


def _written(instruction: CsInsn) -> frozenset[int]:
	"""A call or trap also clobbers the caller-saved registers capstone doesn't list."""
	return frozenset(instruction.regs_access()[1]) | (
		frozenset(ARM_CALLER_SAVED) if instruction.id in _CLOBBERING else frozenset[int]()
	)


def _move_source(instruction: CsInsn) -> int | None:
	operands = tuple(cast("ArmCsOperand", operand) for operand in instruction.operands)
	return (
		operands[1].reg
		if instruction.id in (arm_const.ARM_INS_MOV, arm_const.ARM_INS_MOVS)
		and not arm_predicated(instruction)
		and len(operands) == 2
		and operands[1].type == arm_const.ARM_OP_REG
		and operands[1].shift.type == arm_const.ARM_SFT_INVALID
		and operands[1].reg not in (arm_const.ARM_REG_SP, arm_const.ARM_REG_PC)
		else None
	)
