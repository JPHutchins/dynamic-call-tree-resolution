# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The whole-image and per-thread fixpoints."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from salix import Struct

from dynamic_call_tree_resolution.model import (
	Address,
	CallSite,
	Machine,
)
from dynamic_call_tree_resolution.vsa.abi import (
	normalized,
)
from dynamic_call_tree_resolution.vsa.analysis.creations import (
	Creation,
	modeled,
)
from dynamic_call_tree_resolution.vsa.analysis.seeds import (
	next_round_seeds,
	next_writes,
	observed_seeds,
	seed_from_arguments,
)
from dynamic_call_tree_resolution.vsa.cfg import (
	Block,
	direct_transfer,
	reachable_blocks,
)
from dynamic_call_tree_resolution.vsa.interpret import (
	FunctionResult,
	analyze_function,
)
from dynamic_call_tree_resolution.vsa.lattice import Known
from dynamic_call_tree_resolution.vsa.memory import Context, accumulate_writes, context_for
from dynamic_call_tree_resolution.vsa.state import State, Writes, top_seed

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping
	from concurrent.futures import Executor

	from dynamic_call_tree_resolution.model import (
		Function,
		Program,
		ThreadRoot,
	)
	from dynamic_call_tree_resolution.vsa.escape import Escapes


class ThreadSites(Struct):
	"""What one static thread's own execution reaches, started from its record."""

	reached: frozenset[Address]
	"""The functions the thread runs, up to its calls whose target is unknown."""
	sites: tuple[CallSite, ...]


class Round(Struct):
	"""What one round of the whole-image fixpoint starts from."""

	seeds: Mapping[Address, State]
	writes: Writes


class Rounds(Struct):
	"""What every round of the whole-image fixpoint shares."""

	program: Program
	functions: tuple[tuple[Function, tuple[Block, ...]], ...]
	executor: Executor
	roots: Mapping[Address, State]
	entry_seeds: Mapping[Address, State]
	top: State
	creation: Creation | None
	assumed_unwritten: tuple[tuple[int, int], ...]
	escapes: Escapes


def global_fixpoint(rounds: Rounds, current: Round, round_number: int) -> Round:
	results = tuple(
		rounds.executor.map(
			partial(
				_round_analysis,
				rounds.program,
				context_for(
					rounds.program,
					current.writes,
					assumed_unwritten=rounds.assumed_unwritten,
					escapes=rounds.escapes,
				),
				current.seeds,
			),
			_reached(rounds.functions, current.seeds, rounds.program.machine),
		)
	)
	following = Round(
		seeds={
			**next_round_seeds(
				current.seeds,
				observed_seeds(
					modeled(
						[observation for result in results for observation in result.observations],
						rounds.creation,
					),
					rounds.roots,
					rounds.entry_seeds,
					rounds.program,
				),
				rounds.top,
				round_number,
			),
			**rounds.roots,
		},
		writes=next_writes(
			current.writes,
			accumulate_writes(current.writes, [result.writes for result in results]),
			round_number,
		),
	)
	return current if following == current else global_fixpoint(rounds, following, round_number + 1)


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


def final_result(
	program: Program,
	context: Context,
	seeds: Mapping[Address, State],
	function_blocks: tuple[Function, tuple[Block, ...]],
) -> FunctionResult:
	function, blocks = function_blocks
	return analyze_function(
		context,
		function,
		blocks,
		seeds.get(normalized(function.address, program.machine), top_seed(program.machine)),
	)


def thread_sites(
	program: Program,
	functions: tuple[tuple[Function, tuple[Block, ...]], ...],
	context: Context,
	executor: Executor,
	trampoline: Address,
	thread: ThreadRoot,
) -> ThreadSites:
	start = {
		trampoline: seed_from_arguments(
			{
				position: Known(values=frozenset({value}))
				for position, value in enumerate((thread.entry, *thread.arguments))
			},
			program,
		)
	}
	seeds = _thread_fixpoint(program, functions, context, executor, start, start, 1)
	reached = _reached(functions, seeds, program.machine)
	return ThreadSites(
		reached=frozenset(function.address for function, _ in reached),
		sites=tuple(
			site
			for result in executor.map(partial(final_result, program, context, seeds), reached)
			for site in result.sites
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
	observed = observed_seeds(
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
	next_seeds = next_round_seeds(seeds, observed, top_seed(program.machine), round_number)
	return (
		seeds
		if next_seeds == seeds
		else _thread_fixpoint(
			program, functions, context, executor, start, next_seeds, round_number + 1
		)
	)


def analysis_roots(
	program: Program,
	functions: tuple[tuple[Function, tuple[Block, ...]], ...],
	taken: frozenset[Address],
) -> frozenset[Address]:
	machine = program.machine
	starts = frozenset(normalized(function.address, machine) for function, _ in functions)
	reachable = {
		normalized(function.address, machine): reachable_blocks(blocks)
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
	seeded = (
		(starts - transferred)
		| gap_targets
		| frozenset(normalized(address, machine) for address in taken)
	) & starts
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
