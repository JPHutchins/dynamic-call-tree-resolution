# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Evaluating Arm operands against known registers and the image."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from capstone import arm_const

from dynamic_call_tree_resolution.model import Address
from dynamic_call_tree_resolution.points_to import (
	in_writable_memory,
	instruction_set_at,
	memory_at,
)
from dynamic_call_tree_resolution.vsa.abi import (
	program_counter,
)

if TYPE_CHECKING:
	from collections.abc import Mapping

	from capstone import ArmCsOperand, CsInsn

	from dynamic_call_tree_resolution.model import Program


WORD_MASK: Final = 0xFFFF_FFFF


def pc_relative(program: Program, instruction: CsInsn) -> int:
	return program_counter(instruction, instruction_set_at(program, Address(instruction.address)))


def operand_sum(
	registers: Mapping[int, int], augend: ArmCsOperand, addend: ArmCsOperand
) -> int | None:
	left = operand_value(registers, augend)
	right = operand_value(registers, addend)
	return None if left is None or right is None else left + right


def operand_value(registers: Mapping[int, int], operand: ArmCsOperand) -> int | None:
	return (
		operand.imm
		if operand.type == arm_const.ARM_OP_IMM
		else _shifted(registers.get(operand.reg), operand)
		if operand.type == arm_const.ARM_OP_REG
		else None
	)


def _shifted(value: int | None, operand: ArmCsOperand) -> int | None:
	if value is None:
		return None
	match operand.shift.type:
		case arm_const.ARM_SFT_INVALID:
			return value
		case arm_const.ARM_SFT_LSL:
			return value << operand.shift.value
		case _:
			return None


def memory_operand_base(
	program: Program, instruction: CsInsn, registers: Mapping[int, int], operand: ArmCsOperand
) -> int | None:
	return (
		pc_relative(program, instruction)
		if operand.mem.base == arm_const.ARM_REG_PC
		else registers.get(operand.mem.base)
	)


def memory_operand_address(
	registers: Mapping[int, int], operand: ArmCsOperand, base: int | None
) -> int | None:
	index = 0 if operand.mem.index == 0 else _shifted(registers.get(operand.mem.index), operand)
	return (
		None
		if base is None or index is None or operand.subtracted
		else (base + index + operand.mem.disp) & WORD_MASK
	)


def read_memory(program: Program, address: int, width: int) -> int | None:
	data = memory_at(program, Address(address), width)
	return (
		int.from_bytes(data, program.byte_order)
		if len(data) == width and not in_writable_memory(program, Address(address))
		else None
	)


def arm_operands(instruction: CsInsn) -> tuple[ArmCsOperand, ...]:
	return tuple(cast("ArmCsOperand", operand) for operand in instruction.operands)
