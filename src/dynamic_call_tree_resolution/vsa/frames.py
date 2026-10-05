# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Stack depths measured from machine code, for code without a ``.su`` record (#195)."""

from __future__ import annotations

from functools import cache, partial
from itertools import accumulate, groupby
from typing import TYPE_CHECKING, assert_never

from salix import Struct, replace

from dynamic_call_tree_resolution.model import Address, Machine
from dynamic_call_tree_resolution.vsa.abi import SP_REGISTERS, disassemblers, normalized
from dynamic_call_tree_resolution.vsa.cfg import (
	branch_target,
	call_target,
	function_graph,
	indirect_operand,
)
from dynamic_call_tree_resolution.vsa.interpret import block_states, transfer
from dynamic_call_tree_resolution.vsa.memory import context_for
from dynamic_call_tree_resolution.vsa.state import NO_WRITES, top_seed

if TYPE_CHECKING:
	from collections.abc import Callable, Mapping
	from collections.abc import Set as AbstractSet

	from capstone import Cs, CsInsn

	from dynamic_call_tree_resolution.model import Function, InstructionSet, Program
	from dynamic_call_tree_resolution.vsa.cfg import Block
	from dynamic_call_tree_resolution.vsa.memory import Context
	from dynamic_call_tree_resolution.vsa.state import State


class _Own(Struct):
	"""How deep a function's own code takes the stack, and how deep it is at each call."""

	depth: int
	calls: tuple[tuple[int, Address], ...]


class _Callee(Struct):
	"""A direct call or branch into another function."""

	start: Address


class _Lost(Struct):
	"""A transfer whose target the code does not name, or names outside every function."""


type _Exit = _Callee | _Lost | None


def code_depths(program: Program, starts: frozenset[Address]) -> Mapping[Address, int]:
	"""How deep each function takes the stack, its callees included, where its code can say."""
	match program.machine:
		case Machine.EM_ARM:
			measure = cache(
				partial(
					_own,
					program,
					disassemblers(),
					context_for(program, NO_WRITES),
					_functions_by_start(program),
				)
			)
			return {
				start: depth
				for start in starts
				for depth in (_worst(measure, start, frozenset()),)
				if depth is not None
			}
		case Machine.EM_386 | Machine.EM_X86_64:
			return {}
		case _ as unreachable:
			assert_never(unreachable)


def _worst(
	measure: Callable[[Address], _Own | None], start: Address, visiting: frozenset[Address]
) -> int | None:
	match measure(start):
		case None:
			return None
		case _Own(depth=depth, calls=calls):
			return _deepest(measure, depth, calls, visiting | {start})
		case _ as unreachable:
			assert_never(unreachable)


def _deepest(
	measure: Callable[[Address], _Own | None],
	deepest: int,
	calls: tuple[tuple[int, Address], ...],
	visiting: frozenset[Address],
) -> int | None:
	return deepest if not calls else _through(measure, deepest, calls[0], calls[1:], visiting)


def _through(
	measure: Callable[[Address], _Own | None],
	deepest: int,
	call: tuple[int, Address],
	rest: tuple[tuple[int, Address], ...],
	visiting: frozenset[Address],
) -> int | None:
	at, callee = call
	match None if callee in visiting else _worst(measure, callee, visiting):
		case None:
			return None
		case callee_depth:
			return _deepest(measure, max(deepest, at + callee_depth), rest, visiting)


def _own(
	program: Program,
	decoders: Mapping[InstructionSet, Cs],
	context: Context,
	functions: Mapping[Address, Function],
	start: Address,
) -> _Own | None:
	match (
		function_graph(
			program,
			decoders,
			start,
			replace(functions[start], size=_extent(program, functions, start)),
		)
		if start in functions
		else None
	):
		case None:
			return None
		case (_, blocks):
			return _measured_blocks(program, context, blocks, functions.keys())
		case _ as unreachable:
			assert_never(unreachable)


def _measured_blocks(
	program: Program, context: Context, blocks: tuple[Block, ...], starts: AbstractSet[Address]
) -> _Own | None:
	addresses = frozenset(
		instruction.address for block in blocks for instruction in block.instructions
	)
	return (
		None
		if any(
			_lost(_exit(program, instruction, addresses, starts))
			for block in blocks
			for instruction in block.instructions
		)
		else _measured(
			program,
			_steps(context, blocks, block_states(context, blocks, top_seed(program.machine))[0]),
			addresses,
			starts,
		)
	)


def _extent(program: Program, functions: Mapping[Address, Function], start: Address) -> int:
	return (
		functions[start].size
		or min(
			(
				*(following for following in functions if following > start),
				*(
					base + len(section.data)
					for base, section in program.sections.items()
					if base <= start < base + len(section.data)
				),
			),
			default=start,
		)
		- start
	)


def _functions_by_start(program: Program) -> Mapping[Address, Function]:
	return {
		start: max(group, key=_size)
		for start, group in groupby(
			sorted(program.functions.values(), key=partial(_code_start, program.machine)),
			key=partial(_code_start, program.machine),
		)
	}


def _code_start(machine: Machine, function: Function) -> Address:
	return normalized(function.address, machine)


def _size(function: Function) -> int:
	return function.size


def _step(context: Context, state: State, instruction: CsInsn) -> State:
	return transfer(context, instruction, state)


def _steps(
	context: Context, blocks: tuple[Block, ...], states: Mapping[Address, State]
) -> tuple[tuple[CsInsn, State], ...]:
	return tuple(
		step
		for block in blocks
		if block.start in states
		for step in zip(
			block.instructions,
			accumulate(
				block.instructions[:-1], partial(_step, context), initial=states[block.start]
			),
			strict=True,
		)
	)


def _measured(
	program: Program,
	steps: tuple[tuple[CsInsn, State], ...],
	addresses: frozenset[int],
	starts: AbstractSet[Address],
) -> _Own | None:
	offsets = tuple(state.sp_offsets.get(SP_REGISTERS[program.machine][0]) for _, state in steps)
	exits = tuple(_exit(program, instruction, addresses, starts) for instruction, _ in steps)
	return (
		_Own(
			depth=max((_depth(offset) for offset in offsets if offset is not None), default=0),
			calls=tuple(
				(_depth(offset), callee)
				for offset, exit_ in zip(offsets, exits, strict=True)
				if offset is not None
				for callee in (_called(exit_),)
				if callee is not None
			),
		)
		if None not in offsets
		else None
	)


def _depth(offsets: frozenset[int]) -> int:
	return max(0, -min(offsets, default=0))


def _exit(
	program: Program,
	instruction: CsInsn,
	addresses: frozenset[int],
	starts: AbstractSet[Address],
) -> _Exit:
	if indirect_operand(instruction, program.machine) is not None:
		return _Lost()
	branch = branch_target(instruction, program.machine)
	called = call_target(instruction, program.machine)
	match (
		called
		if called is not None
		else branch.target
		if branch is not None and branch.target not in addresses
		else None
	):
		case None:
			return None
		case target:
			callee = normalized(Address(target), program.machine)
			return _Callee(start=callee) if callee in starts else _Lost()


def _called(exit_: _Exit) -> Address | None:
	match exit_:
		case _Callee(start=start):
			return start
		case _Lost() | None:
			return None
		case _ as unreachable:
			assert_never(unreachable)


def _lost(exit_: _Exit) -> bool:
	match exit_:
		case _Lost():
			return True
		case _Callee() | None:
			return False
		case _ as unreachable:
			assert_never(unreachable)
