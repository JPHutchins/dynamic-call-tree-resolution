# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Analyzing every function of an image."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial, reduce
from itertools import groupby
from typing import TYPE_CHECKING, assert_never, cast

from salix import Struct

from dynamic_call_tree_resolution.model import (
	BARE_METAL,
	Address,
	CallSite,
	InstructionFamily,
	Machine,
)
from dynamic_call_tree_resolution.vsa.abi import (
	is_data,
	normalized,
)
from dynamic_call_tree_resolution.vsa.analysis.creations import (
	functions_named,
	thread_creation,
)
from dynamic_call_tree_resolution.vsa.analysis.rounds import (
	Round,
	Rounds,
	ThreadSites,
	analysis_roots,
	final_result,
	global_fixpoint,
	thread_sites,
)
from dynamic_call_tree_resolution.vsa.analysis.seeds import (
	seed_from_thread,
)
from dynamic_call_tree_resolution.vsa.cfg import (
	FunctionGraph,
	control_flow_graphs,
)
from dynamic_call_tree_resolution.vsa.escape import escapes
from dynamic_call_tree_resolution.vsa.fallback import address_taken, referenced_only_at, referrers
from dynamic_call_tree_resolution.vsa.memory import Context, context_for
from dynamic_call_tree_resolution.vsa.state import NO_WRITES, join_states, top_seed

if TYPE_CHECKING:
	from collections.abc import Mapping

	from capstone import ArmCsOperand, CsOperand

	from dynamic_call_tree_resolution.model import (
		Program,
		RtosModel,
		ThreadRoot,
	)
	from dynamic_call_tree_resolution.span import Span
	from dynamic_call_tree_resolution.vsa.interpret import CallObservation


class Analysis(Struct):
	"""One image's indirect call sites, and the memory state they were resolved against."""

	sites: tuple[CallSite, ...]
	context: Context
	"""Holds the stores of every function, as the final round saw them."""
	seeded: frozenset[str]
	"""The threads whose entry started from its record's arguments rather than unknown ones."""
	threads: Mapping[str, ThreadSites]
	address_taken: frozenset[Address]
	observations: tuple[CallObservation, ...]
	"""Every call the final pass saw reach a known function, with the arguments it passes."""


def analyze(
	program: Program,
	rtos: RtosModel = BARE_METAL,
	*,
	assumed_unwritten: tuple[Span, ...] = (),
) -> Analysis:
	blocks_by_function = control_flow_graphs(program)
	functions = tuple(blocks_by_function.values())
	_prewarm_instructions(functions, program.machine)
	top = top_seed(program.machine)
	threads_by_entry = {
		entry: tuple(group)
		for entry, group in groupby(sorted(rtos.threads, key=_entry), key=_entry)
	}
	taken = address_taken(program)
	references = referrers(
		program,
		frozenset(threads_by_entry)
		| (
			frozenset(functions_named(program, rtos.trampoline)) & taken
			if rtos.trampoline is not None
			else frozenset[Address]()
		),
	)
	seeded_entries = referenced_only_at(
		references,
		{
			entry: frozenset(thread.entry_slot for thread in group)
			for entry, group in threads_by_entry.items()
		},
	)
	entry_seeds = {
		normalized(entry, program.machine): reduce(
			join_states, (seed_from_thread(thread, program) for thread in threads_by_entry[entry])
		)
		for entry in seeded_entries
	}
	creation = thread_creation(program, rtos, taken, references)
	roots = dict.fromkeys(
		analysis_roots(program, functions, taken)
		- entry_seeds.keys()
		- (frozenset({creation.trampoline}) if creation is not None else frozenset()),
		top,
	)
	with ThreadPoolExecutor() as executor:
		found = escapes(program, functions, executor)
		final = global_fixpoint(
			Rounds(
				program=program,
				functions=functions,
				executor=executor,
				roots=roots,
				entry_seeds=entry_seeds,
				top=top,
				creation=creation,
				assumed_unwritten=assumed_unwritten,
				escapes=found,
			),
			Round(seeds={**entry_seeds, **roots}, writes=NO_WRITES),
			1,
		)
		final_context = context_for(
			program, final.writes, assumed_unwritten=assumed_unwritten, escapes=found
		)
		results = tuple(
			executor.map(partial(final_result, program, final_context, final.seeds), functions)
		)
		return Analysis(
			sites=tuple(site for result in results for site in result.sites),
			context=final_context,
			seeded=frozenset(
				thread.name for entry in seeded_entries for thread in threads_by_entry[entry]
			),
			threads={
				thread.name: thread_sites(
					program, functions, final_context, executor, trampoline, thread
				)
				for trampoline in _trampoline(program, rtos)
				for thread in rtos.threads
			},
			address_taken=taken,
			observations=tuple(
				observation
				for result in results
				for observation in (*result.observations, *result.indirect_observations)
			),
		)


def _entry(thread: ThreadRoot) -> Address:
	return thread.entry


def _prewarm_instructions(functions: tuple[FunctionGraph, ...], machine: Machine) -> None:
	for graph in functions:
		for block in graph.blocks:
			for instruction in block.instructions:
				if is_data(instruction):
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
