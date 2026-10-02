# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Interprocedural rounds over every function's interpretation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial, reduce
from itertools import groupby
from typing import TYPE_CHECKING, Final, assert_never, cast

from salix import Struct

from dynamic_call_tree_resolution.model import (
	BARE_METAL,
	Address,
	CallSite,
	InstructionFamily,
	Machine,
)
from dynamic_call_tree_resolution.vsa.abi import (
	ARM_ARGUMENT_REGISTERS,
	SP_REGISTERS,
	X86_64_ARGUMENT_REGISTERS,
	normalized,
)
from dynamic_call_tree_resolution.vsa.cfg import Block, control_flow_graphs, direct_transfer
from dynamic_call_tree_resolution.vsa.fallback import address_taken, referenced_only_at
from dynamic_call_tree_resolution.vsa.interpret import (
	CallObservation,
	FunctionResult,
	analyze_function,
)
from dynamic_call_tree_resolution.vsa.lattice import Known, Top, ValueSet
from dynamic_call_tree_resolution.vsa.memory import Context, accumulate_writes, context_for
from dynamic_call_tree_resolution.vsa.state import NO_WRITES, State, Writes, join_states, top_seed

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping
	from concurrent.futures import Executor

	from capstone import ArmCsOperand, CsOperand

	from dynamic_call_tree_resolution.model import Function, Program, RtosModel, ThreadRoot

_WIDENING_ROUND: Final = 8


class ThreadSites(Struct):
	"""What one static thread's own execution reaches, started from its record."""

	reached: frozenset[Address]
	"""The functions the thread runs, up to its calls whose target is unknown."""
	sites: tuple[CallSite, ...]


class Analysis(Struct):
	"""One image's indirect call sites, and the memory state they were resolved against."""

	sites: tuple[CallSite, ...]
	context: Context
	"""Holds the stores of every function, as the final round saw them."""
	seeded: frozenset[str]
	"""The threads whose entry started from its record's arguments rather than unknown ones."""
	threads: Mapping[str, ThreadSites]


def analyze(program: Program, rtos: RtosModel = BARE_METAL) -> Analysis:
	blocks_by_function = control_flow_graphs(program)
	functions = tuple(blocks_by_function.values())
	_prewarm_instructions(functions, program.machine)
	top = top_seed(program.machine)
	threads_by_entry = {
		entry: tuple(group)
		for entry, group in groupby(sorted(rtos.threads, key=_entry), key=_entry)
	}
	seeded_entries = referenced_only_at(
		program,
		{
			entry: frozenset(thread.entry_slot for thread in group)
			for entry, group in threads_by_entry.items()
		},
	)
	entry_seeds = {
		normalized(entry, program.machine): reduce(
			join_states, (_seed_from_thread(thread, program) for thread in threads_by_entry[entry])
		)
		for entry in seeded_entries
	}
	roots = dict.fromkeys(_roots(program, functions) - entry_seeds.keys(), top)
	with ThreadPoolExecutor() as executor:
		final = _global_fixpoint(
			_Rounds(
				program=program,
				functions=functions,
				executor=executor,
				roots=roots,
				entry_seeds=entry_seeds,
				top=top,
			),
			_Round(seeds={**entry_seeds, **roots}, writes=NO_WRITES),
			1,
		)
		final_context = context_for(program, final.writes)
		return Analysis(
			sites=tuple(
				site
				for sites in executor.map(
					partial(_final_sites, program, final_context, final.seeds), functions
				)
				for site in sites
			),
			context=final_context,
			seeded=frozenset(
				thread.name for entry in seeded_entries for thread in threads_by_entry[entry]
			),
			threads={
				thread.name: _thread_sites(
					program, functions, final_context, executor, trampoline, thread
				)
				for trampoline in _trampoline(program, rtos)
				for thread in rtos.threads
			},
		)


class _Round(Struct):
	"""What one round of the whole-image fixpoint starts from."""

	seeds: Mapping[Address, State]
	writes: Writes


class _Rounds(Struct):
	"""What every round of the whole-image fixpoint shares."""

	program: Program
	functions: tuple[tuple[Function, tuple[Block, ...]], ...]
	executor: Executor
	roots: Mapping[Address, State]
	entry_seeds: Mapping[Address, State]
	top: State


def _global_fixpoint(rounds: _Rounds, current: _Round, round_number: int) -> _Round:
	results = tuple(
		rounds.executor.map(
			partial(
				_round_analysis,
				rounds.program,
				context_for(rounds.program, current.writes),
				current.seeds,
			),
			_reached(rounds.functions, current.seeds, rounds.program.machine),
		)
	)
	following = _Round(
		seeds={
			**_next_seeds(
				current.seeds,
				_observed_seeds(
					[observation for result in results for observation in result.observations],
					rounds.roots,
					rounds.entry_seeds,
					rounds.program,
				),
				rounds.top,
				round_number,
			),
			**rounds.roots,
		},
		writes=_next_writes(
			current.writes,
			accumulate_writes(current.writes, [result.writes for result in results]),
			round_number,
		),
	)
	return (
		current if following == current else _global_fixpoint(rounds, following, round_number + 1)
	)


def _next_writes(previous: Writes, accumulated: Writes, round_number: int) -> Writes:
	return accumulated if round_number < _WIDENING_ROUND else _widened_writes(previous, accumulated)


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


def _entry(thread: ThreadRoot) -> Address:
	return thread.entry


def _observed_seeds(
	observations: list[CallObservation],
	roots: Mapping[Address, State],
	entry_seeds: Mapping[Address, State],
	program: Program,
) -> dict[Address, State]:
	return {
		**entry_seeds,
		**{
			callee: join_states(
				entry_seeds.get(callee),
				reduce(
					join_states,
					(_seed_from_observation(observation, program) for observation in group),
				),
			)
			for callee, group in groupby(sorted(observations, key=_callee), key=_callee)
			if callee not in roots
		},
	}


def _next_seeds(
	seeds: Mapping[Address, State],
	observed: Mapping[Address, State],
	top: State,
	round_number: int,
) -> Mapping[Address, State]:
	return observed if round_number < _WIDENING_ROUND else _widened(seeds, observed, top)


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
					_prewarm_operand(operand, machine)


def _prewarm_operand(operand: CsOperand, machine: Machine) -> None:
	match machine.family:
		case InstructionFamily.ARM:
			_ = cast("ArmCsOperand", operand).shift
		case InstructionFamily.X86:
			_ = operand.mem
		case _ as unreachable:
			assert_never(unreachable)


def _trampoline(program: Program, rtos: RtosModel) -> tuple[Address, ...]:
	return tuple(
		normalized(function.address, program.machine)
		for function in program.functions.values()
		if function.name == rtos.trampoline
	)[:1]


def _thread_sites(
	program: Program,
	functions: tuple[tuple[Function, tuple[Block, ...]], ...],
	context: Context,
	executor: Executor,
	trampoline: Address,
	thread: ThreadRoot,
) -> ThreadSites:
	start = {
		trampoline: _seed_from_observation(
			CallObservation(
				callee=trampoline,
				arguments={
					position: Known(values=frozenset({value}))
					for position, value in enumerate((thread.entry, *thread.arguments))
				},
			),
			program,
		)
	}
	seeds = _thread_fixpoint(program, functions, context, executor, start, start, 1)
	reached = _reached(functions, seeds, program.machine)
	return ThreadSites(
		reached=frozenset(function.address for function, _ in reached),
		sites=tuple(
			site
			for sites in executor.map(partial(_final_sites, program, context, seeds), reached)
			for site in sites
		),
	)


def _thread_fixpoint(
	program: Program,
	functions: tuple[tuple[Function, tuple[Block, ...]], ...],
	context: Context,
	executor: Executor,
	start: Mapping[Address, State],
	seeds: Mapping[Address, State],
	round_number: int,
) -> Mapping[Address, State]:
	observed = _observed_seeds(
		[
			observation
			for result in executor.map(
				partial(_round_analysis, program, context, seeds),
				_reached(functions, seeds, program.machine),
			)
			for observation in (*result.observations, *result.indirect_observations)
		],
		{},
		start,
		program,
	)
	next_seeds = _next_seeds(seeds, observed, top_seed(program.machine), round_number)
	return (
		seeds
		if next_seeds == seeds
		else _thread_fixpoint(
			program, functions, context, executor, start, next_seeds, round_number + 1
		)
	)


def _seed_from_thread(thread: ThreadRoot, program: Program) -> State:
	return _seed_from_observation(
		CallObservation(
			callee=thread.entry,
			arguments={
				position: Known(values=frozenset({value}))
				for position, value in enumerate(thread.arguments)
			},
		),
		program,
	)


def _seed_from_observation(observation: CallObservation, program: Program) -> State:
	known = {
		position: values
		for position, value in observation.arguments.items()
		if (values := _known_values(value)) is not None
	}
	match program.machine:
		case Machine.EM_386:
			return _argument_state(
				program,
				{},
				{
					program.pointer_size * (position + 1): values
					for position, values in known.items()
				},
			)
		case Machine.EM_X86_64:
			return _argument_state(
				program,
				{X86_64_ARGUMENT_REGISTERS[position]: values for position, values in known.items()},
				{},
			)
		case Machine.EM_ARM:
			return _argument_state(
				program,
				{ARM_ARGUMENT_REGISTERS[position]: values for position, values in known.items()},
				{},
			)
		case _ as unreachable:
			assert_never(unreachable)


def _known_values(value: ValueSet) -> frozenset[Address] | None:
	match value:
		case Top():
			return None
		case Known(values=values):
			return values
		case _ as unreachable:
			assert_never(unreachable)


def _argument_state(
	program: Program,
	registers: Mapping[int, frozenset[Address]],
	stack: Mapping[int, frozenset[Address]],
) -> State:
	return State(
		registers=registers,
		sp_offsets={SP_REGISTERS[program.machine][0]: frozenset({0})},
		stack=stack,
		globals=NO_WRITES,
		escaped=False,
	)
