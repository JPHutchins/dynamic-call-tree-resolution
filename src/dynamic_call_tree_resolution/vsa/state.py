# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Abstract machine states and their transfer primitives."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, assert_never

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


class State(Struct):
	"""Abstract state at one block entry; an absent entry is Top, an empty set Bottom."""

	registers: Mapping[int, frozenset[Address]]
	sp_offsets: Mapping[int, frozenset[int]]
	"""Copies of the stack pointer, as offsets from its entry value."""
	stack: Mapping[int, frozenset[Address]]
	"""Frame slots, by entry-frame offset."""
	globals: Mapping[Address, frozenset[Address]]
	"""Values written to program-global addresses."""


def entry_state(base: State | None, machine: Machine) -> State:
	if base is None:
		return State(
			registers={},
			sp_offsets={SP_REGISTERS[machine][0]: frozenset({0})},
			stack={},
			globals={},
		)
	return State(registers=base.registers, sp_offsets=base.sp_offsets, stack=base.stack, globals={})


def join_seeds(current: State | None, incoming: State) -> State:
	if current is None:
		return incoming
	return State(
		registers=_adopt_join(current.registers, incoming.registers),
		sp_offsets=current.sp_offsets,
		stack=_adopt_join(current.stack, incoming.stack),
		globals=current.globals,
	)


def _adopt_join(
	current: Mapping[int, frozenset[Address]], incoming: Mapping[int, frozenset[Address]]
) -> dict[int, frozenset[Address]]:
	return {**current, **{key: _adopted(current, key, value) for key, value in incoming.items()}}


def _adopted(
	current: Mapping[int, frozenset[Address]], key: int, incoming: frozenset[Address]
) -> frozenset[Address]:
	if key not in current:
		return incoming
	match join_sets(current[key], incoming):
		case Top():
			return current[key]
		case Known(values=joined):
			return joined
		case _ as unreachable:
			assert_never(unreachable)


def join_states(current: State | None, incoming: State) -> State:
	if current is None:
		return incoming
	return State(
		registers=join_maps(current.registers, incoming.registers),
		sp_offsets=join_maps(current.sp_offsets, incoming.sp_offsets),
		stack=join_maps(current.stack, incoming.stack),
		globals=join_maps(current.globals, incoming.globals),
	)


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
			return dict(mapping)
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
