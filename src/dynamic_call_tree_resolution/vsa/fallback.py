# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Every function whose address the image takes."""

from __future__ import annotations

import re
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Final, assert_never

from capstone import (
	CS_GRP_BRANCH_RELATIVE,
	CS_GRP_CALL,
	CS_GRP_JUMP,
	CS_GRP_RET,
	arm_const,
	x86_const,
)

from dynamic_call_tree_resolution.composition import flatmap
from dynamic_call_tree_resolution.model import (
	Address,
	InstructionFamily,
	InstructionSet,
	ReferenceKind,
	thumb_twin,
)
from dynamic_call_tree_resolution.points_to import instruction_runs, instruction_set_at
from dynamic_call_tree_resolution.vsa.abi import (
	ARM_ADDITION_SIGNS,
	ARM_MOVES,
	disassemblers,
	is_data,
	program_counter,
)
from dynamic_call_tree_resolution.vsa.links import (
	SlotTarget,
	function_extent,
	functions_by_start,
	linked_targets,
)

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator, Mapping
	from collections.abc import Set as AbstractSet

	from capstone import CsInsn, CsOperand

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
		| frozenset(address for address in _computed_addresses(program) if address in functions)
		| linked_address_taken(program)
	)


def linked_address_taken(program: Program) -> frozenset[Address]:
	return frozenset(linked.target for linked in linked_targets(program, ReferenceKind.ADDRESS))


def referrers(
	program: Program, functions: AbstractSet[Address]
) -> Mapping[Address, frozenset[Address]]:
	computed = (
		_slots_by_target(_computed_slots(program), functions)
		if functions
		else dict[Address, frozenset[Address]]()
	)
	linked = _slots_by_target(linked_targets(program, ReferenceKind.ADDRESS), functions)
	relocated = _slots_by_target(
		(
			SlotTarget(slot=relocation.slot, target=relocation.target)
			for relocation in program.relocations
		),
		functions,
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
	references: Iterable[SlotTarget], targets: AbstractSet[Address]
) -> Mapping[Address, frozenset[Address]]:
	return {
		target: frozenset(reference.slot for reference in group)
		for target, group in groupby(
			sorted(
				(reference for reference in references if reference.target in targets),
				key=_target,
			),
			key=_target,
		)
	}


def _target(reference: SlotTarget) -> Address:
	return reference.target


def _occurrences(program: Program, function: Address) -> Iterator[Address]:
	pattern = re.compile(
		b"(?=" + re.escape(function.to_bytes(program.pointer_size, program.byte_order)) + b")"
	)
	return flatmap(partial(_section_occurrences, program, pattern), program.sections)


def _section_occurrences(
	program: Program, pattern: re.Pattern[bytes], start: Address
) -> Iterator[Address]:
	return (
		Address(start + found.start()) for found in pattern.finditer(program.sections[start].data)
	)


def _computed_addresses(program: Program) -> Iterator[Address]:
	return (computed.target for computed in _computed_slots(program))


def _computed_slots(program: Program) -> Iterator[SlotTarget]:
	return flatmap(partial(_function_slots, program), _non_branch_instructions(program))


def _function_slots(program: Program, instructions: tuple[CsInsn, ...]) -> Iterator[SlotTarget]:
	return flatmap(partial(_instruction_slots, program, instructions), range(len(instructions)))


def _instruction_slots(
	program: Program, instructions: tuple[CsInsn, ...], index: int
) -> Iterator[SlotTarget]:
	slot = Address(instructions[index].address)
	return (
		SlotTarget(slot=slot, target=Address(value))
		for value in _instruction_addresses(program, instructions, index)
	)


def _non_branch_instructions(program: Program) -> Iterator[tuple[CsInsn, ...]]:
	decoders = disassemblers(program.arm_core)
	functions = functions_by_start(program)
	return (
		tuple(
			filter(
				_is_non_branch,
				flatmap(
					lambda run: decoders[instruction_set_at(program, run.start)].disasm(
						run.code, run.start
					),
					instruction_runs(program, start, function_extent(program, functions, start)),
				),
			)
		)
		for start in functions
	)


def _is_non_branch(instruction: CsInsn) -> bool:
	return not is_data(instruction) and not any(
		instruction.group(group) for group in _BRANCH_GROUPS
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
	match instruction.id, operands:
		case arm_const.ARM_INS_ADR, [_, offset]:
			yield _pc_relative(program, instruction, offset.imm)
		case addition, [_, base, offset] if (
			addition in ARM_ADDITION_SIGNS and base.reg == arm_const.ARM_REG_PC
		):
			yield _pc_relative(program, instruction, ARM_ADDITION_SIGNS[addition] * offset.imm)
		case arm_const.ARM_INS_MOVT, [destination, high]:
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
			if earlier.id in ARM_MOVES
			and earlier.operands[0].reg == register
			and earlier.operands[1].type == arm_const.ARM_OP_IMM
		),
		None,
	)
	if low is not None:
		yield (high << 16) | (low & 0xFFFF)


def _x86_addresses(program: Program, instruction: CsInsn) -> Iterator[int]:
	addresses = (
		_x86_operand_address(program, instruction, operand) for operand in instruction.operands
	)
	return (address for address in addresses if address is not None)


def _x86_operand_address(program: Program, instruction: CsInsn, operand: CsOperand) -> int | None:
	match operand.type:
		case x86_const.X86_OP_IMM:
			return operand.imm
		case x86_const.X86_OP_MEM if (
			instruction.mnemonic == "lea" and operand.mem.base == x86_const.X86_REG_RIP
		):
			return (
				program_counter(
					instruction, instruction_set_at(program, Address(instruction.address))
				)
				+ operand.mem.disp
			)
		case _:
			return None
