# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Control-flow graphs of every function, cut at calls that never return."""

from __future__ import annotations

from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, assert_never

from capstone import arm_const
from salix import Struct, replace

from dynamic_call_tree_resolution.model import Address, InstructionFamily, Machine
from dynamic_call_tree_resolution.vsa.abi import (
	ARM_CALLS,
	arm_predicated,
	disassemblers,
	normalized,
)
from dynamic_call_tree_resolution.vsa.cfg.blocks import (
	Block,
	CallsNoReturn,
	FunctionGraph,
	conditional_exit,
	function_graph,
	reachable_blocks,
)
from dynamic_call_tree_resolution.vsa.cfg.jump_tables import (
	dispatches_indirectly,
)
from dynamic_call_tree_resolution.vsa.cfg.operands import (
	WORD_MASK,
	arm_operands,
	pc_relative,
	read_memory,
)
from dynamic_call_tree_resolution.vsa.cfg.transfers import (
	RegisterSite,
	branch_target,
	call_target,
	indirect_operand,
)
from dynamic_call_tree_resolution.vsa.links import function_extent, functions_by_start

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping

	from capstone import Cs, CsInsn

	from dynamic_call_tree_resolution.model import Function, InstructionSet, Program


def _function_address(function: Function) -> Address:
	return function.address


def control_flow_graphs(program: Program) -> dict[Address, FunctionGraph]:
	decoders = disassemblers(program.arm_core)
	by_start = functions_by_start(program)
	return _cut_calls_that_never_return(
		program,
		{
			start: graph
			for start, functions in groupby(
				sorted(program.functions.values(), key=_function_address),
				key=partial(_code_start, program.machine),
			)
			if (
				graph := _first_graph(
					program, decoders, start, _with_extent(program, by_start, start, functions)
				)
			)
			is not None
		},
	)


def _with_extent(
	program: Program,
	by_start: Mapping[Address, Function],
	start: Address,
	functions: Iterable[Function],
) -> Iterable[Function]:
	return (
		functions
		if by_start[start].size
		else (replace(by_start[start], size=function_extent(program, by_start, start)),)
	)


def _code_start(machine: Machine, function: Function) -> Address:
	return normalized(function.address, machine)


def _first_graph(
	program: Program,
	decoders: Mapping[InstructionSet, Cs],
	start: Address,
	functions: Iterable[Function],
) -> FunctionGraph | None:
	return next(
		filter(
			None, (function_graph(program, decoders, start, function) for function in functions)
		),
		None,
	)


class _Exits(Struct):
	calls: Mapping[Address, Address]
	"""The function each block that ends in an unconditional direct call calls, by block."""
	exits: Mapping[Address, Address | None]
	"""The function each block that can leave the function leaves into; None when it may return."""
	dispatches: bool


def _cut_calls_that_never_return(
	program: Program, graphs: Mapping[Address, FunctionGraph]
) -> dict[Address, FunctionGraph]:
	exits = {start: _exits(program, graph.blocks) for start, graph in graphs.items()}
	never_returning = _never_returning(graphs, exits, program.never_returns)
	return {
		start: FunctionGraph(
			graph.function,
			tuple(
				replace(block, condition=CallsNoReturn())
				if exits[start].calls.get(block.start) in never_returning
				else block
				for block in graph.blocks
			),
		)
		for start, graph in graphs.items()
	}


def _never_returning(
	graphs: Mapping[Address, FunctionGraph],
	exits: Mapping[Address, _Exits],
	known: frozenset[Address],
) -> frozenset[Address]:
	grown = known | frozenset(
		start
		for start, graph in graphs.items()
		if start not in known and _leaves_only_into(graph.blocks, exits[start], known)
	)
	return known if grown == known else _never_returning(graphs, exits, grown)


def _leaves_only_into(
	blocks: tuple[Block, ...], exits: _Exits, never_returning: frozenset[Address]
) -> bool:
	reached = reachable_blocks(
		blocks,
		frozenset(start for start, callee in exits.calls.items() if callee in never_returning),
	)
	return not exits.dispatches and all(
		target is not None and target in never_returning
		for start, target in exits.exits.items()
		if start in reached
	)


def _exits(program: Program, blocks: tuple[Block, ...]) -> _Exits:
	machine = program.machine
	starts = frozenset(block.start for block in blocks)
	by_start = {block.start: block for block in blocks}
	predecessors = {
		start: tuple(source for source, _ in group)
		for start, group in groupby(
			sorted((block.start, successor) for block in blocks for successor in block.successors),
			key=_successor,
		)
	}
	return _Exits(
		calls={
			block.start: normalized(Address(callee), machine)
			for block in blocks
			if (callee := call_target(block.instructions[-1], machine)) is not None
			and not _predicated(block.instructions[-1], machine)
		},
		exits={
			block.start: _left_into(
				program, _straight_line(machine, block, by_start, predecessors, frozenset())
			)
			for block in blocks
			if _leaves(block, machine, starts)
		},
		dispatches=any(dispatches_indirectly(block.instructions[-1], machine) for block in blocks),
	)


def _successor(edge: tuple[Address, Address]) -> Address:
	return edge[1]


def _straight_line(
	machine: Machine,
	block: Block,
	by_start: Mapping[Address, Block],
	predecessors: Mapping[Address, tuple[Address, ...]],
	visited: frozenset[Address],
) -> tuple[CsInsn, ...]:
	"""The block's instructions, after those of the blocks that only run straight into it."""
	match predecessors.get(block.start, ()):
		case (source,) if (
			source not in visited
			and by_start[source].successors == (block.start,)
			and call_target(by_start[source].instructions[-1], machine) is None
			and indirect_operand(by_start[source].instructions[-1], machine) is None
		):
			return (
				*_straight_line(
					machine, by_start[source], by_start, predecessors, visited | {block.start}
				),
				*block.instructions,
			)
		case _:
			return block.instructions


def _predicated(instruction: CsInsn, machine: Machine) -> bool:
	match machine.family:
		case InstructionFamily.X86:
			return False
		case InstructionFamily.ARM:
			return arm_predicated(instruction)
		case _ as unreachable:
			assert_never(unreachable)


def _leaves(block: Block, machine: Machine, starts: frozenset[Address]) -> bool:
	branch = branch_target(block.instructions[-1], machine)
	return (
		not block.successors
		or (branch is not None and branch.conditional and Address(branch.target) not in starts)
		or (branch is None and conditional_exit(block.instructions[-1], machine, None))
	)


def _left_into(program: Program, instructions: tuple[CsInsn, ...]) -> Address | None:
	last = instructions[-1]
	machine = program.machine
	callee = call_target(last, machine)
	branch = branch_target(last, machine)
	return (
		(normalized(Address(callee), machine) if not _predicated(last, machine) else None)
		if callee is not None
		else normalized(Address(branch.target), machine)
		if branch is not None
		else _literal_jump(program, instructions)
	)


def _literal_jump(program: Program, instructions: tuple[CsInsn, ...]) -> Address | None:
	match program.machine.family, indirect_operand(instructions[-1], program.machine):
		case InstructionFamily.ARM, RegisterSite(operand=operand) if (
			instructions[-1].id not in ARM_CALLS
		):
			target = next(
				(
					_literal(program, instruction)
					for instruction in reversed(instructions[:-1])
					if operand.reg in instruction.regs_access()[1]
				),
				None,
			)
			return None if target is None else normalized(Address(target), program.machine)
		case _:
			return None


def _literal(program: Program, instruction: CsInsn) -> int | None:
	operands = arm_operands(instruction)
	return (
		read_memory(
			program,
			(pc_relative(program, instruction) + operands[1].mem.disp) & WORD_MASK,
			program.pointer_size,
		)
		if instruction.id == arm_const.ARM_INS_LDR
		and len(operands) == 2
		and operands[1].type == arm_const.ARM_OP_MEM
		and operands[1].mem.base == arm_const.ARM_REG_PC
		and operands[1].mem.index == 0
		else None
	)
