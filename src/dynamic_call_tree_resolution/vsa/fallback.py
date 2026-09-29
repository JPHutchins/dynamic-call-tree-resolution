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
	Cs,
	arm_const,
	x86_const,
)

from dynamic_call_tree_resolution.model import Address, Machine, thumb_twin
from dynamic_call_tree_resolution.points_to import memory_at
from dynamic_call_tree_resolution.vsa.abi import DISASSEMBLERS, normalized

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
	disassembler = Cs(*DISASSEMBLERS[program.machine])
	disassembler.detail = True
	disassembler.skipdata = True
	for function in program.functions.values():
		start = normalized(function.address, program.machine)
		instructions = tuple(
			instruction
			for instruction in disassembler.disasm(memory_at(program, start, function.size), start)
			if instruction.id != 0 and not any(instruction.group(group) for group in _BRANCH_GROUPS)
		)
		for index, instruction in enumerate(instructions):
			yield from (
				_arm_addresses(instruction, instructions[:index])
				if program.machine is Machine.EM_ARM
				else _x86_addresses(instruction)
			)


_BRANCH_GROUPS: Final = (CS_GRP_CALL, CS_GRP_JUMP, CS_GRP_RET, CS_GRP_BRANCH_RELATIVE)


def _arm_addresses(instruction: CsInsn, preceding: tuple[CsInsn, ...]) -> Iterator[int]:
	operands = instruction.operands
	yield from (operand.imm for operand in operands if operand.type == arm_const.ARM_OP_IMM)
	match instruction.mnemonic.split(".")[0], operands:
		case "adr", [_, offset]:
			yield thumb_twin(Address(_aligned_pc(instruction) + offset.imm))
		case (("add" | "addw"), [_, base, offset]) if base.reg == arm_const.ARM_REG_PC:
			yield thumb_twin(Address(_aligned_pc(instruction) + offset.imm))
		case (("sub" | "subw"), [_, base, offset]) if base.reg == arm_const.ARM_REG_PC:
			yield thumb_twin(Address(_aligned_pc(instruction) - offset.imm))
		case "movt", [destination, high]:
			yield from _movt_addresses(high.imm, destination.reg, preceding)
		case _:
			pass


def _aligned_pc(instruction: CsInsn) -> int:
	return (instruction.address + 4) & ~3


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


def _x86_addresses(instruction: CsInsn) -> Iterator[int]:
	for operand in instruction.operands:
		if operand.type == x86_const.X86_OP_IMM:
			yield operand.imm
		if (
			instruction.mnemonic == "lea"
			and operand.type == x86_const.X86_OP_MEM
			and operand.mem.base == x86_const.X86_REG_RIP
		):
			yield instruction.address + instruction.size + operand.mem.disp
