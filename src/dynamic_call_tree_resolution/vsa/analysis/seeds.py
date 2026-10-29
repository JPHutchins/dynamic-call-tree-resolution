# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Seeding callees' parameters from their callers' arguments, widened across rounds."""

from __future__ import annotations

from functools import reduce
from itertools import groupby
from typing import TYPE_CHECKING, Final, assert_never

from dynamic_call_tree_resolution.model import (
	Address,
	Machine,
)
from dynamic_call_tree_resolution.vsa.abi import (
	ARM_ARGUMENT_REGISTERS,
	SP_REGISTERS,
	X86_64_ARGUMENT_REGISTERS,
)
from dynamic_call_tree_resolution.vsa.lattice import Known, Top, ValueSet, is_top, join
from dynamic_call_tree_resolution.vsa.state import (
	NO_WRITES,
	UNESCAPED,
	State,
	Writes,
	join_states,
)

if TYPE_CHECKING:
	from collections.abc import Mapping

	from dynamic_call_tree_resolution.model import (
		Program,
		ThreadRoot,
	)
	from dynamic_call_tree_resolution.vsa.interpret import CallObservation


_WIDENING_ROUND: Final = 8


def observed_seeds(
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
					(seed_from_arguments(observation.arguments, program) for observation in group),
				),
			)
			for callee, group in groupby(sorted(observations, key=_callee), key=_callee)
			if callee not in roots
		},
	}


def _callee(observation: CallObservation) -> Address:
	return observation.callee


def observed_entry_words(
	observations: list[CallObservation],
	roots: Mapping[Address, State],
	entry_seeds: Mapping[Address, State],
) -> dict[Address, ValueSet]:
	return {
		site: word
		for callee, group in groupby(sorted(observations, key=_callee), key=_callee)
		if callee not in roots and callee not in entry_seeds
		for observed in (tuple(group),)
		for site in sorted({site for observation in observed for site in observation.entry_words})
		for word in (
			reduce(join, (observation.entry_words.get(site, Top()) for observation in observed)),
		)
		if not is_top(word)
	}


def seed_from_arguments(arguments: Mapping[int, ValueSet], program: Program) -> State:
	known = {
		position: values
		for position, value in arguments.items()
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
				{
					ARM_ARGUMENT_REGISTERS[position]: values
					for position, values in known.items()
					if position < len(ARM_ARGUMENT_REGISTERS)
				},
				{
					program.pointer_size * (position - len(ARM_ARGUMENT_REGISTERS)): values
					for position, values in known.items()
					if position >= len(ARM_ARGUMENT_REGISTERS)
				},
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
		escaped=UNESCAPED,
		loaded_from={},
		strided={},
	)


def seed_from_thread(thread: ThreadRoot, program: Program) -> State:
	return seed_from_arguments(
		{
			position: Known(values=frozenset({value}))
			for position, value in enumerate(thread.arguments)
		},
		program,
	)


def next_round_seeds(
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


def next_entry_words(
	words: Mapping[Address, ValueSet], observed: Mapping[Address, ValueSet], round_number: int
) -> Mapping[Address, ValueSet]:
	return (
		observed
		if round_number < _WIDENING_ROUND
		else {site: word for site, word in observed.items() if words.get(site) == word}
	)


def _widened_writes(previous: Writes, current: Writes) -> Writes:
	return Writes(
		values={
			address: value
			if address not in previous.values or previous.values[address] == value
			else Top()
			for address, value in current.values.items()
		},
		unknown_store=current.unknown_store,
		clobbered=current.clobbered,
	)


def next_writes(previous: Writes, accumulated: Writes, round_number: int) -> Writes:
	return accumulated if round_number < _WIDENING_ROUND else _widened_writes(previous, accumulated)
