# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Every function whose address the image takes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from capstone import (
	CS_GRP_BRANCH_RELATIVE,
	CS_GRP_CALL,
	CS_GRP_JUMP,
	CS_GRP_RET,
	arm_const,
	x86_const,
)

from dynamic_call_tree_resolution.model import Address, InstructionSet, Machine, thumb_twin
from dynamic_call_tree_resolution.points_to import instruction_runs, instruction_set_at
from dynamic_call_tree_resolution.vsa.abi import disassemblers, normalized, program_counter

if TYPE_CHECKING:
	from collections.abc import Iterator

	from capstone import CsInsn

	from dynamic_call_tree_resolution.model import Program


def address_taken(program: Program) -> frozenset[Address]:
	functions = frozenset(program.functions)
	return frozenset(
		address
		for address in functions
		if any(
			address.to_bytes(program.pointer_size, program.byte_order) in section.data
			for section in program.sections.values()
		)
	) | frozenset(Address(value) for value in _computed_addresses(program) if value in functions)


def _computed_addresses(program: Program) -> Iterator[int]:
	decoders = disassemblers()
	for function in program.functions.values():
		start = normalized(function.address, program.machine)
		instructions = tuple(
			instruction
			for address, code in instruction_runs(program, start, function.size)
			for instruction in decoders[instruction_set_at(program, address)].disasm(code, address)
			if instruction.id != 0 and not any(instruction.group(group) for group in _BRANCH_GROUPS)
		)
		for index, instruction in enumerate(instructions):
			yield from (
				_arm_addresses(program, instruction, instructions[:index])
				if program.machine is Machine.EM_ARM
				else _x86_addresses(program, instruction)
			)


_BRANCH_GROUPS: Final = (CS_GRP_CALL, CS_GRP_JUMP, CS_GRP_RET, CS_GRP_BRANCH_RELATIVE)


def _arm_addresses(
	program: Program, instruction: CsInsn, preceding: tuple[CsInsn, ...]
) -> Iterator[int]:
	operands = instruction.operands
	yield from (operand.imm for operand in operands if operand.type == arm_const.ARM_OP_IMM)
	match instruction.mnemonic.split(".")[0], operands:
		case "adr", [_, offset]:
			yield _pc_relative(program, instruction, offset.imm)
		case (("add" | "addw"), [_, base, offset]) if base.reg == arm_const.ARM_REG_PC:
			yield _pc_relative(program, instruction, offset.imm)
		case (("sub" | "subw"), [_, base, offset]) if base.reg == arm_const.ARM_REG_PC:
			yield _pc_relative(program, instruction, -offset.imm)
		case "movt", [destination, high]:
			yield from _movt_addresses(high.imm, destination.reg, preceding)
		case _:
			pass


def _pc_relative(program: Program, instruction: CsInsn, offset: int) -> Address:
	instruction_set = instruction_set_at(program, Address(instruction.address))
	target = Address(program_counter(instruction, instruction_set) + offset)
	return thumb_twin(target) if instruction_set is InstructionSet.T32 else target


def _movt_addresses(high: int, register: int, preceding: tuple[CsInsn, ...]) -> Iterator[int]:
	low = next(
		(
			earlier.operands[1].imm
			for earlier in reversed(preceding)
			if earlier.mnemonic.split(".")[0] == "movw" and earlier.operands[0].reg == register
		),
		None,
	)
	if low is not None:
		yield (high << 16) | (low & 0xFFFF)


def _x86_addresses(program: Program, instruction: CsInsn) -> Iterator[int]:
	for operand in instruction.operands:
		if operand.type == x86_const.X86_OP_IMM:
			yield operand.imm
		if (
			instruction.mnemonic == "lea"
			and operand.type == x86_const.X86_OP_MEM
			and operand.mem.base == x86_const.X86_REG_RIP
		):
			yield (
				program_counter(
					instruction, instruction_set_at(program, Address(instruction.address))
				)
				+ operand.mem.disp
			)
