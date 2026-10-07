# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Every function whose address the image takes."""

from __future__ import annotations

import re
from itertools import groupby, repeat
from typing import TYPE_CHECKING, Final, assert_never

from capstone import (
	CS_GRP_BRANCH_RELATIVE,
	CS_GRP_CALL,
	CS_GRP_JUMP,
	CS_GRP_RET,
	arm_const,
	x86_const,
)

from dynamic_call_tree_resolution.model import (
	Address,
	InstructionFamily,
	InstructionSet,
	ReferenceKind,
	thumb_twin,
)
from dynamic_call_tree_resolution.points_to import instruction_runs, instruction_set_at
from dynamic_call_tree_resolution.vsa.abi import (
	arm_mnemonic,
	disassemblers,
	program_counter,
)
from dynamic_call_tree_resolution.vsa.links import (
	function_extent,
	functions_by_start,
	linked_targets,
)

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator, Mapping
	from collections.abc import Set as AbstractSet

	from capstone import CsInsn

	from dynamic_call_tree_resolution.model import Program


def address_taken(program: Program) -> frozenset[Address]:
	functions = frozenset(program.functions)
	return (
		frozenset(
			address
			for address in functions
			if any(
				address.to_bytes(program.pointer_size, program.byte_order) in section.data
				for section in program.sections.values()
			)
		)
		| frozenset(Address(value) for value in _computed_addresses(program) if value in functions)
		| linked_address_taken(program)
	)


def linked_address_taken(program: Program) -> frozenset[Address]:
	return frozenset(function for _, function in linked_targets(program, ReferenceKind.ADDRESS))


def referrers(
	program: Program, functions: AbstractSet[Address]
) -> Mapping[Address, frozenset[Address]]:
	computed = (
		_slots_by_target(_computed_slots(program), functions)
		if functions
		else dict[int, frozenset[Address]]()
	)
	linked = _slots_by_target(linked_targets(program, ReferenceKind.ADDRESS), functions)
	relocated = _slots_by_target(
		((relocation.slot, relocation.target) for relocation in program.relocations), functions
	)
	return {
		function: frozenset(_occurrences(program, function))
		| computed.get(function, frozenset())
		| linked.get(function, frozenset())
		| relocated.get(function, frozenset())
		for function in functions
	}


def referenced_only_at(
	references: Mapping[Address, frozenset[Address]],
	slots_by_function: Mapping[Address, frozenset[Address]],
) -> frozenset[Address]:
	return frozenset(
		function for function, slots in slots_by_function.items() if references[function] <= slots
	)


def _slots_by_target(
	references: Iterable[tuple[Address, int]], targets: AbstractSet[Address]
) -> Mapping[int, frozenset[Address]]:
	return {
		target: frozenset(slot for slot, _ in group)
		for target, group in groupby(
			sorted(
				((slot, target) for slot, target in references if target in targets),
				key=_target,
			),
			key=_target,
		)
	}


def _target(reference: tuple[Address, int]) -> int:
	return reference[1]


def _occurrences(program: Program, function: Address) -> Iterator[Address]:
	pattern = re.compile(
		b"(?=" + re.escape(function.to_bytes(program.pointer_size, program.byte_order)) + b")"
	)
	return (
		Address(start + match.start())
		for start, section in program.sections.items()
		for match in pattern.finditer(section.data)
	)


def _computed_addresses(program: Program) -> Iterator[int]:
	for instructions in _non_branch_instructions(program):
		for index in range(len(instructions)):
			yield from _instruction_addresses(program, instructions, index)


def _computed_slots(program: Program) -> Iterator[tuple[Address, int]]:
	for instructions in _non_branch_instructions(program):
		for index in range(len(instructions)):
			yield from zip(
				repeat(Address(instructions[index].address)),
				_instruction_addresses(program, instructions, index),
			)


def _non_branch_instructions(program: Program) -> Iterator[tuple[CsInsn, ...]]:
	decoders = disassemblers(program.arm_core)
	functions = functions_by_start(program)
	return (
		tuple(
			instruction
			for address, code in instruction_runs(
				program, start, function_extent(program, functions, start)
			)
			for instruction in decoders[instruction_set_at(program, address)].disasm(code, address)
			if instruction.id != 0 and not any(instruction.group(group) for group in _BRANCH_GROUPS)
		)
		for start in functions
	)


def _instruction_addresses(
	program: Program, instructions: tuple[CsInsn, ...], index: int
) -> Iterator[int]:
	match program.machine.family:
		case InstructionFamily.ARM:
			return _arm_addresses(program, instructions[index], instructions[:index])
		case InstructionFamily.X86:
			return _x86_addresses(program, instructions[index])
		case _ as unreachable:
			assert_never(unreachable)


_BRANCH_GROUPS: Final = (CS_GRP_CALL, CS_GRP_JUMP, CS_GRP_RET, CS_GRP_BRANCH_RELATIVE)


def _arm_addresses(
	program: Program, instruction: CsInsn, preceding: tuple[CsInsn, ...]
) -> Iterator[int]:
	operands = instruction.operands
	yield from (operand.imm for operand in operands if operand.type == arm_const.ARM_OP_IMM)
	match arm_mnemonic(instruction), operands:
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
	target = Address(
		program_counter(instruction, instruction_set_at(program, Address(instruction.address)))
		+ offset
	)
	return (
		thumb_twin(target) if instruction_set_at(program, target) is InstructionSet.T32 else target
	)


def _movt_addresses(high: int, register: int, preceding: tuple[CsInsn, ...]) -> Iterator[int]:
	low = next(
		(
			earlier.operands[1].imm
			for earlier in reversed(preceding)
			if arm_mnemonic(earlier) == "movw" and earlier.operands[0].reg == register
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
