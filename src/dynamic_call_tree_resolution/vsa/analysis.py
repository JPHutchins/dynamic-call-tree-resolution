# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Interprocedural rounds over every function's interpretation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial, reduce
from itertools import groupby
from typing import TYPE_CHECKING, Final, assert_never, cast

from salix import Struct

from dynamic_call_tree_resolution.model import Address, CallSite, Machine
from dynamic_call_tree_resolution.vsa.abi import (
	ARM_ARGUMENT_REGISTERS,
	SP_REGISTERS,
	X86_64_ARGUMENT_REGISTERS,
	normalized,
)
from dynamic_call_tree_resolution.vsa.cfg import Block, control_flow_graphs, direct_transfer
from dynamic_call_tree_resolution.vsa.fallback import address_taken
from dynamic_call_tree_resolution.vsa.interpret import (
	CallObservation,
	FunctionResult,
	analyze_function,
)
from dynamic_call_tree_resolution.vsa.lattice import Known, Top
from dynamic_call_tree_resolution.vsa.memory import Context, accumulate_writes, context_for
from dynamic_call_tree_resolution.vsa.state import NO_WRITES, State, Writes, join_states, top_seed

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping

	from capstone import ArmCsOperand

	from dynamic_call_tree_resolution.model import Function, Program

_WIDENING_ROUND: Final = 8


class Analysis(Struct):
	"""One image's indirect call sites, and the memory state they were resolved against."""

	sites: tuple[CallSite, ...]
	context: Context
	"""Holds the stores of every function, as the final round saw them."""


def analyze(program: Program) -> Analysis:
	blocks_by_function = control_flow_graphs(program)
	functions = tuple(blocks_by_function.values())
	_prewarm_instructions(functions, program.machine)
	top = top_seed(program.machine)
	roots = dict.fromkeys(_roots(program, functions), top)
	seeds: Mapping[Address, State] = roots
	global_writes = NO_WRITES
	round_number = 0
	with ThreadPoolExecutor() as executor:
		while True:
			round_number += 1
			context = context_for(program, global_writes)
			observations: list[CallObservation] = []
			written: list[Writes] = []
			for result in executor.map(
				partial(_round_analysis, program, context, seeds),
				_reached(functions, seeds, program.machine),
			):
				written.append(result.writes)
				observations.extend(result.observations)
			observed = _observed_seeds(observations, roots, program)
			next_seeds = {
				**(observed if round_number < _WIDENING_ROUND else _widened(seeds, observed, top)),
				**roots,
			}
			accumulated = accumulate_writes(global_writes, written)
			next_writes = (
				accumulated
				if round_number < _WIDENING_ROUND
				else _widened_writes(global_writes, accumulated)
			)
			if next_seeds == seeds and next_writes == global_writes:
				break
			seeds, global_writes = next_seeds, next_writes
		final_context = context_for(program, global_writes)
		return Analysis(
			sites=tuple(
				site
				for sites in executor.map(
					partial(_final_sites, program, final_context, seeds), functions
				)
				for site in sites
			),
			context=final_context,
		)


def _roots(
	program: Program, functions: tuple[tuple[Function, tuple[Block, ...]], ...]
) -> frozenset[Address]:
	machine = program.machine
	starts = frozenset(normalized(function.address, machine) for function, _ in functions)
	reachable = {
		normalized(function.address, machine): _reachable_blocks(blocks)
		for function, blocks in functions
	}
	transferred = frozenset[Address]().union(
		*(
			_callees(blocks, machine, normalized(function.address, machine), starts)
			for function, blocks in functions
		)
	)
	gap_targets = frozenset[Address]().union(
		*(
			_callees(
				(
					block
					for block in blocks
					if block.start not in reachable[normalized(function.address, machine)]
				),
				machine,
				normalized(function.address, machine),
				starts,
			)
			for function, blocks in functions
		)
	)
	taken = frozenset(normalized(address, machine) for address in address_taken(program))
	seeded = ((starts - transferred) | gap_targets | taken) & starts
	callees = {
		normalized(function.address, machine): _callees(
			(
				block
				for block in blocks
				if block.start in reachable[normalized(function.address, machine)]
			),
			machine,
			normalized(function.address, machine),
			starts,
		)
		for function, blocks in functions
	}
	return seeded | (starts - _call_closure(seeded, callees))


def _callees(
	blocks: Iterable[Block], machine: Machine, own_start: Address, starts: frozenset[Address]
) -> frozenset[Address]:
	return frozenset(
		transfer.target
		for block in blocks
		if (transfer := direct_transfer(block.instructions[-1], machine, own_start, starts))
		is not None
	)


def _reachable_blocks(blocks: tuple[Block, ...]) -> frozenset[Address]:
	by_start = {block.start: block for block in blocks}
	seen = {blocks[0].start}
	stack = [blocks[0].start]
	while stack:
		for successor in by_start[stack.pop()].successors:
			if successor not in seen:
				seen.add(successor)
				stack.append(successor)
	return frozenset(seen)


def _call_closure(
	roots: frozenset[Address], callees: Mapping[Address, frozenset[Address]]
) -> frozenset[Address]:
	seen = set(roots)
	stack = list(roots)
	while stack:
		for callee in callees.get(stack.pop(), frozenset()):
			if callee not in seen:
				seen.add(callee)
				stack.append(callee)
	return frozenset(seen)


def _reached(
	functions: tuple[tuple[Function, tuple[Block, ...]], ...],
	seeds: Mapping[Address, State],
	machine: Machine,
) -> tuple[tuple[Function, tuple[Block, ...]], ...]:
	return tuple(
		function_blocks
		for function_blocks in functions
		if normalized(function_blocks[0].address, machine) in seeds
	)


def _callee(observation: CallObservation) -> Address:
	return observation.callee


def _observed_seeds(
	observations: list[CallObservation], roots: Mapping[Address, State], program: Program
) -> dict[Address, State]:
	return {
		callee: reduce(
			join_states, (_seed_from_observation(observation, program) for observation in group)
		)
		for callee, group in groupby(sorted(observations, key=_callee), key=_callee)
		if callee not in roots
	}


def _widened(
	previous: Mapping[Address, State], observed: Mapping[Address, State], top: State
) -> dict[Address, State]:
	return {
		**previous,
		**{
			callee: seed if callee not in previous or previous[callee] == seed else top
			for callee, seed in observed.items()
		},
	}


def _widened_writes(previous: Writes, current: Writes) -> Writes:
	return Writes(
		values={
			address: value
			if address not in previous.values or previous.values[address] == value
			else Top()
			for address, value in current.values.items()
		},
		wild=current.wild,
	)


def _round_analysis(
	program: Program,
	context: Context,
	seeds: Mapping[Address, State],
	function_blocks: tuple[Function, tuple[Block, ...]],
) -> FunctionResult:
	function, blocks = function_blocks
	return analyze_function(
		context, function, blocks, seeds[normalized(function.address, program.machine)]
	)


def _final_sites(
	program: Program,
	context: Context,
	seeds: Mapping[Address, State],
	function_blocks: tuple[Function, tuple[Block, ...]],
) -> tuple[CallSite, ...]:
	function, blocks = function_blocks
	return analyze_function(
		context, function, blocks, seeds[normalized(function.address, program.machine)]
	).sites


def _prewarm_instructions(
	functions: tuple[tuple[Function, tuple[Block, ...]], ...], machine: Machine
) -> None:
	for _, blocks in functions:
		for block in blocks:
			for instruction in block.instructions:
				if instruction.mnemonic == ".byte":
					continue
				for operand in instruction.operands:
					if machine is Machine.EM_ARM:
						_ = cast("ArmCsOperand", operand).shift
					else:
						_ = operand.mem


def _seed_from_observation(observation: CallObservation, program: Program) -> State:
	registers: dict[int, frozenset[Address]] = {}
	stack: dict[int, frozenset[Address]] = {}
	for position, value in observation.arguments.items():
		match value:
			case Top():
				continue
			case Known(values=values):
				if program.machine is Machine.EM_386:
					stack[program.pointer_size * (position + 1)] = values
				elif program.machine is Machine.EM_X86_64:
					registers[X86_64_ARGUMENT_REGISTERS[position]] = values
				else:
					registers[ARM_ARGUMENT_REGISTERS[position]] = values
			case _ as unreachable:
				assert_never(unreachable)
	return State(
		registers=registers,
		sp_offsets={SP_REGISTERS[program.machine][0]: frozenset({0})},
		stack=stack,
		globals=NO_WRITES,
		escaped=False,
	)
