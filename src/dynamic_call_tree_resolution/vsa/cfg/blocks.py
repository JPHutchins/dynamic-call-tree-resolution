# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""A function's basic blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from salix import Struct, replace

from dynamic_call_tree_resolution.model import Address, InstructionFamily, Machine
from dynamic_call_tree_resolution.points_to import (
	instruction_runs,
	instruction_set_at,
)
from dynamic_call_tree_resolution.span import Span
from dynamic_call_tree_resolution.vsa.abi import (
	X86_CALLS,
	arm_predicated,
	is_data,
)
from dynamic_call_tree_resolution.vsa.cfg.jump_tables import (
	JumpTable,
	dispatches_indirectly,
	jump_table,
)
from dynamic_call_tree_resolution.vsa.cfg.narrowing import narrowing
from dynamic_call_tree_resolution.vsa.cfg.transfers import (
	BranchTarget,
	arm_falls_through,
	block_ends,
	branch_target,
	call_target,
	indirect_operand,
	is_control_transfer,
	is_returning_trap,
)
from dynamic_call_tree_resolution.vsa.conditions import Conditional, arm_branch_test

if TYPE_CHECKING:
	from collections.abc import Mapping

	from capstone import Cs, CsInsn

	from dynamic_call_tree_resolution.model import Function, InstructionSet, Program


class Block(Struct):
	"""One basic block of a function's code."""

	start: Address
	instructions: tuple[CsInsn, ...]
	successors: tuple[Address, ...]
	condition: Conditional | CallsNoReturn | None


class CallsNoReturn(Struct):
	"""A block-ending direct call to a function that never returns."""


def reachable_blocks(
	blocks: tuple[Block, ...], cut: frozenset[Address] = frozenset()
) -> frozenset[Address]:
	by_start = {block.start: block for block in blocks}
	seen = {blocks[0].start}
	stack = [blocks[0].start]
	while stack:
		start = stack.pop()
		for successor in () if start in cut else by_start[start].successors:
			if successor not in seen:
				seen.add(successor)
				stack.append(successor)
	return frozenset(seen)


def function_graph(
	program: Program, decoders: Mapping[InstructionSet, Cs], start: Address, function: Function
) -> tuple[Function, tuple[Block, ...]] | None:
	runs = instruction_runs(program, start, function.size)
	if not any(run.code for run in runs):
		return None
	decoded = tuple(
		_decoded(
			decoders[instruction_set_at(program, run.start)],
			program,
			run.start,
			run.code,
			Span(start=Address(start), end=Address(start + function.size)),
		)
		for run in runs
	)
	return (
		function,
		_build_blocks(
			tuple(instruction for run, _ in decoded for instruction in run),
			program.machine,
			{address: table for _, tables in decoded for address, table in tables.items()},
		),
	)


def _decoded(
	disassembler: Cs, program: Program, start: int, code: bytes, function: Span
) -> tuple[tuple[CsInsn, ...], Mapping[int, JumpTable | None]]:
	instructions: list[CsInsn] = []
	dispatches: dict[int, JumpTable | None] = {}
	offset = 0
	while offset < len(code):
		for instruction in disassembler.disasm(code[offset:], start + offset):
			instructions.append(instruction)
			if dispatches_indirectly(instruction, program.machine):
				table = jump_table(instructions, program, function)
				dispatches[instruction.address] = table
				if table is not None and table.skip is not None:
					offset = table.skip - start
					break
		else:
			break
	return tuple(instructions), dispatches


def _build_blocks(
	instructions: tuple[CsInsn, ...],
	machine: Machine,
	dispatches: Mapping[int, JumpTable | None],
) -> tuple[Block, ...]:
	by_address = {instruction.address: index for index, instruction in enumerate(instructions)}
	branched_to = frozenset(
		branch.target
		for instruction in instructions
		if (branch := branch_target(instruction, machine)) is not None
	)
	entered = branched_to | {
		target for table in dispatches.values() if table is not None for target in table.targets
	}
	cases = {
		address: table.targets
		for address, table in dispatches.items()
		if table is not None
		and entered.isdisjoint(table.guarded)
		and all(target in by_address for target in table.targets)
	}
	targets = branched_to | {target for targets in cases.values() for target in targets}
	blocks: list[Block] = []
	current: list[CsInsn] = []
	for index, instruction in enumerate(instructions):
		if (
			instruction.address in targets
			or indirect_operand(instruction, machine) is not None
			or call_target(instruction, machine) is not None
			or is_returning_trap(instruction, machine)
			or (
				(branch := branch_target(instruction, machine)) is not None
				and branch.target not in by_address
			)
		) and current:
			blocks.append(
				Block(
					start=Address(current[0].address),
					instructions=tuple(current),
					successors=(Address(instruction.address),),
					condition=None,
				)
			)
			current = []
		current.append(instruction)
		if block_ends(instruction, machine):
			next_address = (
				instructions[index + 1].address if index + 1 < len(instructions) else None
			)
			blocks.append(
				Block(
					start=Address(current[0].address),
					instructions=tuple(current),
					successors=cases[instruction.address]
					if instruction.address in cases
					else ()
					if instruction.address in dispatches
					else _block_end_successors(instruction, machine, by_address, next_address),
					condition=None
					if instruction.address in cases or instruction.address in dispatches
					else _condition(tuple(current), machine, by_address, next_address),
				)
			)
			current = []
	if current:
		blocks.append(
			Block(
				start=Address(current[0].address),
				instructions=tuple(current),
				successors=(),
				condition=None,
			)
		)
	every_start = tuple(block.start for block in blocks)
	return tuple(
		replace(block, successors=every_start)
		if block.instructions[-1].address in dispatches
		and block.instructions[-1].address not in cases
		else block
		for block in blocks
	)


def _block_end_successors(
	instruction: CsInsn, machine: Machine, by_address: Mapping[int, int], next_address: int | None
) -> tuple[Address, ...]:
	branch = branch_target(instruction, machine)
	if branch is not None:
		return (
			*((Address(next_address),) if branch.conditional and next_address is not None else ()),
			*((Address(branch.target),) if branch.target in by_address else ()),
		)
	fallthrough = (Address(next_address),) if next_address is not None else ()
	if is_data(instruction):
		return fallthrough
	return fallthrough if _transfer_returns(instruction, machine) else ()


def _transfer_returns(instruction: CsInsn, machine: Machine) -> bool:
	match machine.family:
		case InstructionFamily.X86:
			return instruction.mnemonic in X86_CALLS or is_returning_trap(instruction, machine)
		case InstructionFamily.ARM:
			return arm_falls_through(instruction, machine) or arm_predicated(instruction)
		case _ as unreachable:
			assert_never(unreachable)


def _condition(
	instructions: tuple[CsInsn, ...],
	machine: Machine,
	by_address: Mapping[int, int],
	next_address: int | None,
) -> Conditional | None:
	branch = branch_target(instructions[-1], machine)
	test = (
		arm_branch_test(instructions)
		if conditional_exit(instructions[-1], machine, branch)
		else None
	)
	return (
		None
		if test is None
		else Conditional(
			test=test,
			taken=(Address(branch.target),)
			if branch is not None and branch.target in by_address
			else (),
			fallthrough=(Address(next_address),) if next_address is not None else (),
			narrowing=narrowing(instructions, test),
		)
	)


def conditional_exit(instruction: CsInsn, machine: Machine, branch: BranchTarget | None) -> bool:
	match machine.family:
		case InstructionFamily.X86:
			return False
		case InstructionFamily.ARM:
			return (
				branch.conditional
				if branch is not None
				else arm_predicated(instruction)
				and is_control_transfer(instruction, machine)
				and not arm_falls_through(instruction, machine)
			)
		case _ as unreachable:
			assert_never(unreachable)
