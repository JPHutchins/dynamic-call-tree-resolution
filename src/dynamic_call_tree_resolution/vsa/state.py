# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Abstract machine states and their transfer primitives."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Final, assert_never

from salix import Struct

from dynamic_call_tree_resolution.vsa.abi import SP_REGISTERS
from dynamic_call_tree_resolution.vsa.lattice import (
	Known,
	Lattice,
	OffsetSet,
	Top,
	ValueSet,
	bind,
	capped,
	join,
	join_maps,
	join_sets,
	lookup,
	map_entry,
	put_value,
)

if TYPE_CHECKING:
	from collections.abc import Mapping

	from capstone import CsInsn

	from dynamic_call_tree_resolution.model import Address, Machine


class Writes(Struct):
	"""What the analyzed stores wrote to program-global memory."""

	values: Mapping[Address, ValueSet]
	"""By address; an address no store wrote is absent."""
	wild: bool
	"""Some store's address was unknown, so any writable address may hold anything."""


NO_WRITES: Final = Writes(values={}, wild=False)


class State(Struct):
	"""Abstract state at one block entry; an absent entry is Top, an empty set Bottom."""

	registers: Mapping[int, frozenset[Address]]
	sp_offsets: Mapping[int, frozenset[int]]
	"""Copies of the stack pointer, as offsets from its entry value."""
	stack: Mapping[int, frozenset[Address]]
	"""Frame slots, by entry-frame offset."""
	globals: Writes
	"""What the stores on the path to the block wrote."""


def top_seed(machine: Machine) -> State:
	return State(
		registers={},
		sp_offsets={SP_REGISTERS[machine][0]: frozenset({0})},
		stack={},
		globals=NO_WRITES,
	)


def join_states(current: State | None, incoming: State) -> State:
	if current is None:
		return incoming
	return State(
		registers=join_maps(current.registers, incoming.registers),
		sp_offsets=join_maps(current.sp_offsets, incoming.sp_offsets),
		stack=join_maps(current.stack, incoming.stack),
		globals=_join_writes(current.globals, incoming.globals),
	)


def _join_writes(current: Writes, incoming: Writes) -> Writes:
	return Writes(
		values={
			address: join(current.values[address], incoming.values[address])
			for address in current.values.keys() & incoming.values.keys()
		},
		wild=current.wild or incoming.wild,
	)


def store_global(writes: Writes, addresses: ValueSet, value: ValueSet) -> Writes:
	match addresses:
		case Top():
			return Writes(values={}, wild=True)
		case Known(values=targets):
			return Writes(
				values={
					**writes.values,
					**{
						address: join(writes.values[address], value)
						if address in writes.values
						else value
						for address in targets
					},
				},
				wild=writes.wild,
			)
		case _ as unreachable:
			assert_never(unreachable)


def set_register(state: State, register: int, value: ValueSet) -> State:
	sp_offsets = dict(state.sp_offsets)
	sp_offsets.pop(register, None)
	return State(
		registers=put_value(state.registers, register, value),
		sp_offsets=sp_offsets,
		stack=state.stack,
		globals=state.globals,
	)


def set_offsets(state: State, register: int, value: OffsetSet) -> State:
	registers = dict(state.registers)
	registers.pop(register, None)
	return State(
		registers=registers,
		sp_offsets=put_value(state.sp_offsets, register, value),
		stack=state.stack,
		globals=state.globals,
	)


def top_registers(state: State, registers: tuple[int, ...]) -> State:
	remaining = {
		register: value for register, value in state.registers.items() if register not in registers
	}
	return State(
		registers=remaining, sp_offsets=state.sp_offsets, stack=state.stack, globals=state.globals
	)


def top_written(instruction: CsInsn, state: State) -> State:
	written = set(instruction.regs_access()[1])
	return State(
		registers={
			register: value
			for register, value in state.registers.items()
			if register not in written
		},
		sp_offsets={
			register: value
			for register, value in state.sp_offsets.items()
			if register not in written
		},
		stack=state.stack,
		globals=state.globals,
	)


def stack_read(stack: Mapping[int, frozenset[Address]], offsets: OffsetSet) -> ValueSet:
	return bind(offsets, partial(_frame_read, stack))


def _frame_read(stack: Mapping[int, frozenset[Address]], offsets: frozenset[int]) -> ValueSet:
	if any(offset not in stack for offset in offsets):
		return Top()
	return capped(frozenset(value for offset in offsets for value in stack[offset]))


def union_write[K: int](
	mapping: Mapping[K, frozenset[Address]], keys: Lattice[K], value: ValueSet
) -> dict[K, frozenset[Address]]:
	match keys:
		case Top():
			return {}
		case Known(values=written):
			return {
				**{key: values for key, values in mapping.items() if key not in written},
				**dict(
					entry
					for key in written
					for entry in map_entry(key, _updated(mapping, key, value))
				),
			}
		case _ as unreachable:
			assert_never(unreachable)


def _updated[K: int](mapping: Mapping[K, frozenset[Address]], key: K, value: ValueSet) -> ValueSet:
	return bind(
		value,
		lambda values: join_sets(mapping[key], values) if key in mapping else Known(values=values),
	)


def copy_register(state: State, destination: int, source: int) -> State:
	if destination == source:
		return state
	if source in state.sp_offsets:
		return set_offsets(state, destination, Known(values=state.sp_offsets[source]))
	return set_register(state, destination, lookup(state.registers, source))
