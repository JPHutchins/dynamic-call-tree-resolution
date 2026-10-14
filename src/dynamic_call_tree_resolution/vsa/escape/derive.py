# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The addresses and constants an instruction puts in the registers it writes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from capstone import CS_AC_WRITE, arm_const

from dynamic_call_tree_resolution.model import Address
from dynamic_call_tree_resolution.points_to import instruction_set_at, memory_at
from dynamic_call_tree_resolution.vsa.abi import (
	ARM_ADDITION_SIGNS,
	ARM_MULTIPLE_STORES,
	ARM_REGISTER_BYTES,
	ARM_STORE_WIDTHS,
	program_counter,
)
from dynamic_call_tree_resolution.vsa.escape.values import (
	NOTHING,
	WORD_MASK,
	Derived,
	Held,
	Registers,
	offset_by,
	size,
	union,
	unknown_offset,
)

if TYPE_CHECKING:
	from collections.abc import Mapping

	from capstone import ArmCsOperand, CsInsn

	from dynamic_call_tree_resolution.model import Program
	from dynamic_call_tree_resolution.vsa.escape.image import Image


MOVES: Final = frozenset({arm_const.ARM_INS_MOV, arm_const.ARM_INS_MOVS})


LISTED_BASES: Final = frozenset(
	{
		arm_const.ARM_INS_LDM,
		arm_const.ARM_INS_LDMDA,
		arm_const.ARM_INS_LDMDB,
		arm_const.ARM_INS_LDMIB,
		arm_const.ARM_INS_STM,
		arm_const.ARM_INS_STMDA,
		arm_const.ARM_INS_STMDB,
		arm_const.ARM_INS_STMIB,
		arm_const.ARM_INS_VLDMIA,
		arm_const.ARM_INS_VLDMDB,
		arm_const.ARM_INS_VSTMIA,
		arm_const.ARM_INS_VSTMDB,
	}
)


_DESCENDING: Final = frozenset(
	{
		arm_const.ARM_INS_LDMDA,
		arm_const.ARM_INS_LDMDB,
		arm_const.ARM_INS_STMDA,
		arm_const.ARM_INS_STMDB,
		arm_const.ARM_INS_VLDMDB,
		arm_const.ARM_INS_VSTMDB,
	}
)


PLACED_STORES: Final = (frozenset(ARM_STORE_WIDTHS) | ARM_MULTIPLE_STORES) - {
	arm_const.ARM_INS_PUSH,
	arm_const.ARM_INS_VPUSH,
}


def derived_destination(
	image: Image, instruction: CsInsn, operands: tuple[ArmCsOperand, ...], registers: Registers
) -> Mapping[int, tuple[Held, int | None]]:
	if (
		not operands
		or operands[0].type != arm_const.ARM_OP_REG
		or not operands[0].access & CS_AC_WRITE
	):
		return {}
	constant = _constant(image.program, instruction, operands, registers.constants)
	held = _destination_held(image, instruction, operands, registers, constant)
	return {operands[0].reg: (held, constant)} if constant is not None or size(held) else {}


def _constant(
	program: Program,
	instruction: CsInsn,
	operands: tuple[ArmCsOperand, ...],
	constants: Mapping[int, int],
) -> int | None:
	"""The value every path gives the instruction's destination, when the scan can compute it."""
	match instruction.id, operands:
		case arm_const.ARM_INS_LDR, _ if (
			literal := literal_address(program, instruction, operands)
		) is not None:
			return _word(program, literal)
		case arm_const.ARM_INS_ADR, (_, offset):
			return (_pc(program, instruction) + offset.imm) & WORD_MASK
		case identifier, (_, base, offset) if (
			identifier in ARM_ADDITION_SIGNS
			and base.type == arm_const.ARM_OP_REG
			and base.reg == arm_const.ARM_REG_PC
			and offset.type == arm_const.ARM_OP_IMM
		):
			return (
				_pc(program, instruction) + ARM_ADDITION_SIGNS[identifier] * offset.imm
			) & WORD_MASK
		case ((arm_const.ARM_INS_MOV | arm_const.ARM_INS_MOVS), (_, source)):
			return _masked(_operand_constant(source, constants))
		case arm_const.ARM_INS_MOVT, (destination, high) if destination.reg in constants:
			return ((high.imm << 16) | (constants[destination.reg] & 0xFFFF)) & WORD_MASK
		case identifier, _ if identifier in ARM_ADDITION_SIGNS:
			match (
				constants.get(_terms(operands)[0].reg),
				_operand_constant(_terms(operands)[1], constants),
			):
				case int() as augend, int() as addend:
					return (augend + ARM_ADDITION_SIGNS[identifier] * addend) & WORD_MASK
				case _:
					return None
		case _:
			return None


def _masked(value: int | None) -> int | None:
	return None if value is None else value & WORD_MASK


def _word(program: Program, address: int) -> int | None:
	data = memory_at(program, Address(address), 4)
	return int.from_bytes(data, program.byte_order) if len(data) == 4 else None


def _destination_held(
	image: Image,
	instruction: CsInsn,
	operands: tuple[ArmCsOperand, ...],
	registers: Registers,
	constant: int | None,
) -> Held:
	if instruction.id in MOVES and operands[1].type == arm_const.ARM_OP_REG:
		return registers.held.get(operands[1].reg, NOTHING)
	if instruction.id in ARM_ADDITION_SIGNS and _terms(operands)[0].reg != arm_const.ARM_REG_PC:
		return _sum(instruction, operands, registers)
	return (
		Held(
			exact=frozenset({Derived(origin=constant, address=constant, site=instruction.address)})
		)
		if constant is not None and image.nearby[0] <= constant <= image.nearby[1]
		else NOTHING
	)


def _sum(instruction: CsInsn, operands: tuple[ArmCsOperand, ...], registers: Registers) -> Held:
	sign = ARM_ADDITION_SIGNS[instruction.id]
	left, right = _terms(operands)
	if right.type == arm_const.ARM_OP_IMM:
		return offset_by(registers.held.get(left.reg, NOTHING), sign * right.imm)
	return union(
		_offset_or_unknown(
			registers.held.get(left.reg, NOTHING),
			_contribution(registers.constants.get(right.reg), right),
			sign,
		),
		NOTHING
		if sign < 0 and left.reg in registers.constants
		else _offset_or_unknown(
			registers.held.get(right.reg, NOTHING),
			registers.constants.get(left.reg)
			if right.shift.type == arm_const.ARM_SFT_INVALID
			else None,
			sign,
		),
	)


def _offset_or_unknown(held: Held, delta: int | None, sign: int) -> Held:
	return unknown_offset(held) if delta is None else offset_by(held, sign * delta)


def _terms(operands: tuple[ArmCsOperand, ...]) -> tuple[ArmCsOperand, ArmCsOperand]:
	"""An addition's two terms; a two-operand form adds to its destination."""
	return (operands[1], operands[2]) if len(operands) == 3 else (operands[0], operands[1])


def _operand_constant(operand: ArmCsOperand, constants: Mapping[int, int]) -> int | None:
	return (
		operand.imm
		if operand.type == arm_const.ARM_OP_IMM
		else _contribution(constants.get(operand.reg), operand)
		if operand.type == arm_const.ARM_OP_REG
		else None
	)


def _contribution(value: int | None, operand: ArmCsOperand) -> int | None:
	"""What a register operand holding the value adds, after its shift and sign."""
	if value is None or operand.shift.type not in (
		arm_const.ARM_SFT_INVALID,
		arm_const.ARM_SFT_LSL,
	):
		return None
	return (-1 if operand.subtracted else 1) * (
		value << (operand.shift.value if operand.shift.type == arm_const.ARM_SFT_LSL else 0)
	)


def written_back(
	instruction: CsInsn, operands: tuple[ArmCsOperand, ...], registers: Registers
) -> Mapping[int, tuple[Held, int | None]]:
	"""The base register a load or store moves past what it accessed."""
	if instruction.id in LISTED_BASES:
		return (
			{
				operands[0].reg: _moved(
					registers,
					operands[0].reg,
					-listed_width(instruction, operands)
					if instruction.id in _DESCENDING
					else listed_width(instruction, operands),
				)
			}
			if instruction.writeback
			else {}
		)
	memory = next(
		(index for index, operand in enumerate(operands) if operand.type == arm_const.ARM_OP_MEM),
		None,
	)
	if memory is None:
		return {}
	if memory < len(operands) - 1:
		return {
			operands[memory].mem.base: _moved(
				registers,
				operands[memory].mem.base,
				_operand_constant(operands[-1], registers.constants),
			)
		}
	if instruction.writeback:
		return {
			operands[memory].mem.base: _moved(
				registers,
				operands[memory].mem.base,
				None if operands[memory].mem.index else operands[memory].mem.disp,
			)
		}
	return {}


def _moved(registers: Registers, base: int, delta: int | None) -> tuple[Held, int | None]:
	return (
		(unknown_offset(registers.held.get(base, NOTHING)), None)
		if delta is None
		else (
			offset_by(registers.held.get(base, NOTHING), delta),
			_masked(None if base not in registers.constants else registers.constants[base] + delta),
		)
	)


def listed_width(instruction: CsInsn, operands: tuple[ArmCsOperand, ...]) -> int:
	return sum(
		ARM_REGISTER_BYTES.get(instruction.reg_name(operand.reg)[0], 4) for operand in operands[1:]
	)


def store_target(
	instruction: CsInsn, operands: tuple[ArmCsOperand, ...], registers: Registers
) -> Held:
	"""Where a store the analysis can place writes, by what its base register holds."""
	if instruction.id not in PLACED_STORES:
		return NOTHING
	if instruction.id in LISTED_BASES:
		return unknown_offset(
			offset_by(
				registers.held.get(operands[0].reg, NOTHING),
				-listed_width(instruction, operands) if instruction.id in _DESCENDING else 0,
			)
		)
	memory = next(operand.mem for operand in operands if operand.type == arm_const.ARM_OP_MEM)
	return (
		unknown_offset(offset_by(registers.held.get(memory.base, NOTHING), memory.disp))
		if memory.index
		else offset_by(registers.held.get(memory.base, NOTHING), memory.disp)
	)


def literal_address(
	program: Program, instruction: CsInsn, operands: tuple[ArmCsOperand, ...]
) -> int | None:
	"""The word a PC-relative ``ldr`` reads."""
	return (
		(_pc(program, instruction) + operands[1].mem.disp) & WORD_MASK
		if instruction.id == arm_const.ARM_INS_LDR
		and len(operands) == 2
		and operands[1].type == arm_const.ARM_OP_MEM
		and operands[1].mem.base == arm_const.ARM_REG_PC
		and not operands[1].mem.index
		else None
	)


def _pc(program: Program, instruction: CsInsn) -> int:
	return program_counter(instruction, instruction_set_at(program, Address(instruction.address)))


def operands_of(instruction: CsInsn) -> tuple[ArmCsOperand, ...]:
	return tuple(cast("ArmCsOperand", operand) for operand in instruction.operands)
