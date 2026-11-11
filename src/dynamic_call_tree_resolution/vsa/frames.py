# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Stack depths measured from machine code, for code without a ``.su`` record (#195)."""

from __future__ import annotations

from functools import cache, partial
from itertools import accumulate
from typing import TYPE_CHECKING, assert_never

from salix import Struct, replace

from dynamic_call_tree_resolution.composition import flatmap
from dynamic_call_tree_resolution.model import Address, Machine
from dynamic_call_tree_resolution.vsa.abi import (
	ARM_CALLS,
	SP_REGISTERS,
	disassemblers,
	normalized,
)
from dynamic_call_tree_resolution.vsa.cfg import (
	FunctionGraph,
	branch_target,
	call_target,
	every_instruction,
	function_graph,
	indirect_operand,
)
from dynamic_call_tree_resolution.vsa.interpret import block_states, transfer
from dynamic_call_tree_resolution.vsa.links import function_extent, functions_by_start
from dynamic_call_tree_resolution.vsa.memory import context_for
from dynamic_call_tree_resolution.vsa.state import NO_WRITES, top_seed

if TYPE_CHECKING:
	from collections.abc import Callable, Iterator, Mapping
	from collections.abc import Set as AbstractSet

	from capstone import Cs, CsInsn

	from dynamic_call_tree_resolution.model import Function, InstructionSet, Program
	from dynamic_call_tree_resolution.vsa.cfg import Block
	from dynamic_call_tree_resolution.vsa.memory import Context
	from dynamic_call_tree_resolution.vsa.state import State


class _Call(Struct):
	"""A direct call, with the stack depth at it."""

	depth: int
	callee: Address


class _Own(Struct):
	"""How deep a function's own code takes the stack, and how deep it is at each call."""

	depth: int
	calls: tuple[_Call, ...]
	register_calls: int
	"""Calls through a register, whose callees the code does not name."""


class OwnCode(Struct):
	"""A function's own frame, measured from its code, and what that code calls directly."""

	bytes: int
	callees: frozenset[Address]


class _Callee(Struct):
	"""A direct call or branch into another function."""

	start: Address


class _RegisterCall(Struct):
	"""A call through a register: the code goes on after it, to an unnamed callee."""


class _Lost(Struct):
	"""A transfer whose target the code does not name, or names outside every function."""


type _Exit = _Callee | _RegisterCall | _Lost | None


class CodeMeasure(Struct):
	"""One image's code, measured once per function."""

	own: Callable[[Address], _Own | None]


def code_measure(program: Program) -> CodeMeasure:
	match program.machine:
		case Machine.EM_ARM:
			return CodeMeasure(
				own=cache(
					partial(
						_own,
						program,
						disassemblers(program.arm_core),
						context_for(program, NO_WRITES),
						functions_by_start(program),
					)
				)
			)
		case Machine.EM_386 | Machine.EM_X86_64:
			return CodeMeasure(own=_unmeasured)
		case _ as unreachable:
			assert_never(unreachable)


def code_depths(measure: CodeMeasure, starts: frozenset[Address]) -> Mapping[Address, int]:
	"""How deep each function takes the stack, its callees included, where its code can say."""
	depths = {start: _worst(measure.own, start, frozenset()) for start in starts}
	return {start: depth for start, depth in depths.items() if depth is not None}


def own_frames(measure: CodeMeasure, starts: frozenset[Address]) -> Mapping[Address, OwnCode]:
	"""Each function's own frame, where its code can say even though a callee is unnamed."""
	owns = {start: measure.own(start) for start in starts}
	return {
		start: OwnCode(bytes=own.depth, callees=frozenset(call.callee for call in own.calls))
		for start, own in owns.items()
		if own is not None
	}


def _unmeasured(_start: Address) -> _Own | None:
	return None


def _worst(
	measure: Callable[[Address], _Own | None], start: Address, visiting: frozenset[Address]
) -> int | None:
	match measure(start):
		case None:
			return None
		case _Own(register_calls=0, depth=depth, calls=calls):
			return _deepest(measure, depth, calls, visiting | {start})
		case _Own():
			return None
		case _ as unreachable:
			assert_never(unreachable)


def _deepest(
	measure: Callable[[Address], _Own | None],
	deepest: int,
	calls: tuple[_Call, ...],
	visiting: frozenset[Address],
) -> int | None:
	return deepest if not calls else _through(measure, deepest, calls[0], calls[1:], visiting)


def _through(
	measure: Callable[[Address], _Own | None],
	deepest: int,
	call: _Call,
	rest: tuple[_Call, ...],
	visiting: frozenset[Address],
) -> int | None:
	match None if call.callee in visiting else _worst(measure, call.callee, visiting):
		case None:
			return None
		case callee_depth:
			return _deepest(measure, max(deepest, call.depth + callee_depth), rest, visiting)


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
			replace(functions[start], size=function_extent(program, functions, start)),
		)
		if start in functions
		else None
	):
		case None:
			return None
		case FunctionGraph(blocks=blocks):
			return _measured_blocks(program, context, blocks, functions.keys())
		case _ as unreachable:
			assert_never(unreachable)


def _measured_blocks(
	program: Program, context: Context, blocks: tuple[Block, ...], starts: AbstractSet[Address]
) -> _Own | None:
	instructions = tuple(every_instruction(blocks))
	addresses = frozenset(instruction.address for instruction in instructions)
	return (
		None
		if any(
			_lost(_exit(program, instruction, addresses, starts)) for instruction in instructions
		)
		else _measured(
			program,
			_steps(
				context, blocks, block_states(context, blocks, top_seed(program.machine)).in_states
			),
			addresses,
			starts,
		)
	)


class _Step(Struct):
	"""An instruction, with the state before it."""

	instruction: CsInsn
	state: State


def _steps(
	context: Context, blocks: tuple[Block, ...], states: Mapping[Address, State]
) -> tuple[_Step, ...]:
	reached = (block for block in blocks if block.start in states)
	return tuple(flatmap(partial(_block_steps, context, states), reached))


def _block_steps(
	context: Context, states: Mapping[Address, State], block: Block
) -> Iterator[_Step]:
	before_each = accumulate(
		block.instructions[:-1], partial(transfer, context), initial=states[block.start]
	)
	return (
		_Step(instruction, state)
		for instruction, state in zip(block.instructions, before_each, strict=True)
	)


def _measured(
	program: Program,
	steps: tuple[_Step, ...],
	addresses: frozenset[int],
	starts: AbstractSet[Address],
) -> _Own | None:
	offsets = tuple(step.state.sp_offsets.get(SP_REGISTERS[program.machine][0]) for step in steps)
	exits = tuple(_exit(program, step.instruction, addresses, starts) for step in steps)
	calls = (_direct_call(offset, exit_) for offset, exit_ in zip(offsets, exits, strict=True))
	return (
		_Own(
			depth=max((_depth(offset) for offset in offsets if offset is not None), default=0),
			calls=tuple(call for call in calls if call is not None),
			register_calls=sum(map(_through_register, exits)),
		)
		if None not in offsets
		else None
	)


def _direct_call(offsets: frozenset[int] | None, exit_: _Exit) -> _Call | None:
	callee = _called(exit_)
	return (
		_Call(depth=_depth(offsets), callee=callee)
		if offsets is not None and callee is not None
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
		return _RegisterCall() if instruction.id in ARM_CALLS else _Lost()
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
		case _RegisterCall() | _Lost() | None:
			return None
		case _ as unreachable:
			assert_never(unreachable)


def _through_register(exit_: _Exit) -> bool:
	match exit_:
		case _RegisterCall():
			return True
		case _Callee() | _Lost() | None:
			return False
		case _ as unreachable:
			assert_never(unreachable)


def _lost(exit_: _Exit) -> bool:
	match exit_:
		case _Lost():
			return True
		case _Callee() | _RegisterCall() | None:
			return False
		case _ as unreachable:
			assert_never(unreachable)
