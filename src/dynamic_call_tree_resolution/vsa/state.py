# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Abstract machine states and their transfer primitives."""

from __future__ import annotations

from functools import partial, reduce
from typing import TYPE_CHECKING, Final, assert_never

from capstone import CS_OP_REG
from salix import Struct

from dynamic_call_tree_resolution.model import Address
from dynamic_call_tree_resolution.vsa.abi import FLAG_REGISTER_NAMES, SP_REGISTERS
from dynamic_call_tree_resolution.vsa.lattice import (
	Known,
	OffsetSet,
	Top,
	ValueSet,
	bind,
	capped,
	is_top,
	join,
	join_maps,
	lookup,
	map_entry,
	put_value,
)

if TYPE_CHECKING:
	from collections.abc import Collection, Container, Iterable, Mapping

	from capstone import CsInsn

	from dynamic_call_tree_resolution.model import Machine


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
	escaped: bool
	"""A frame address may be held where the analysis does not track it."""
	loaded_from: Mapping[int, Address]
	"""The one image address each register's value was loaded from."""


class Store(Struct):
	"""What one store writes from each address it may start at."""

	words: tuple[ValueSet, ...]
	"""Pointer-sized values from the start up; the written bytes past them hold Top."""
	width: int
	"""In bytes."""


def top_seed(machine: Machine) -> State:
	return State(
		registers={},
		sp_offsets={SP_REGISTERS[machine][0]: frozenset({0})},
		stack={},
		globals=NO_WRITES,
		escaped=False,
		loaded_from={},
	)


def join_states(current: State | None, incoming: State) -> State:
	if current is None:
		return incoming
	sp_offsets = join_maps(current.sp_offsets, incoming.sp_offsets)
	return State(
		registers=join_maps(current.registers, incoming.registers),
		sp_offsets=sp_offsets,
		stack=join_maps(current.stack, incoming.stack),
		globals=_join_writes(current.globals, incoming.globals),
		escaped=current.escaped
		or incoming.escaped
		or frame_copies_lost(current.sp_offsets, incoming.sp_offsets, sp_offsets),
		loaded_from={
			register: address
			for register, address in current.loaded_from.items()
			if incoming.loaded_from.get(register) == address
		},
	)


def frame_copies_lost(
	current: Mapping[int, frozenset[int]],
	incoming: Mapping[int, frozenset[int]],
	joined: Mapping[int, frozenset[int]],
) -> bool:
	return not current.keys() | incoming.keys() <= joined.keys()


def _join_writes(current: Writes, incoming: Writes) -> Writes:
	return Writes(
		values={
			address: join(current.values[address], incoming.values[address])
			for address in current.values.keys() & incoming.values.keys()
		},
		wild=current.wild or incoming.wild,
	)


def set_register(state: State, register: int, value: ValueSet) -> State:
	return set_loaded(state, register, value, None)


def set_loaded(state: State, register: int, value: ValueSet, source: Address | None) -> State:
	sp_offsets = dict(state.sp_offsets)
	sp_offsets.pop(register, None)
	return State(
		registers=put_value(state.registers, register, value),
		sp_offsets=sp_offsets,
		stack=state.stack,
		globals=state.globals,
		escaped=state.escaped,
		loaded_from=_loaded(state.loaded_from, register, source),
	)


def _loaded(
	loaded_from: Mapping[int, Address], register: int, source: Address | None
) -> Mapping[int, Address]:
	return (
		{**loaded_from, register: source}
		if source is not None
		else {held: address for held, address in loaded_from.items() if held != register}
		if register in loaded_from
		else loaded_from
	)


def without_loaded(
	loaded_from: Mapping[int, Address], registers: Collection[int]
) -> Mapping[int, Address]:
	return (
		loaded_from
		if loaded_from.keys().isdisjoint(registers)
		else {
			register: address
			for register, address in loaded_from.items()
			if register not in registers
		}
	)


def set_offsets(state: State, register: int, value: OffsetSet) -> State:
	registers = dict(state.registers)
	registers.pop(register, None)
	return State(
		registers=registers,
		sp_offsets=put_value(state.sp_offsets, register, value),
		stack=state.stack,
		globals=state.globals,
		escaped=state.escaped or is_top(value),
		loaded_from=_loaded(state.loaded_from, register, None),
	)


def top_registers(state: State, registers: tuple[int, ...]) -> State:
	remaining = {
		register: value for register, value in state.registers.items() if register not in registers
	}
	return State(
		registers=remaining,
		sp_offsets=state.sp_offsets,
		stack=state.stack,
		globals=state.globals,
		escaped=state.escaped,
		loaded_from=without_loaded(state.loaded_from, registers),
	)


def frame_based(state: State, machine: Machine, register: int) -> bool:
	return register == SP_REGISTERS[machine][0] or register in state.sp_offsets


def unknown_memory(state: State) -> State:
	return State(
		registers=state.registers,
		sp_offsets=state.sp_offsets,
		stack={},
		globals=Writes(values={}, wild=True),
		escaped=state.escaped,
		loaded_from=state.loaded_from,
	)


def escaping(state: State, registers: Iterable[int]) -> State:
	if state.escaped or not any(register in state.sp_offsets for register in registers):
		return state
	return State(
		registers=state.registers,
		sp_offsets=state.sp_offsets,
		stack=state.stack,
		globals=state.globals,
		escaped=True,
		loaded_from=state.loaded_from,
	)


def top_written(instruction: CsInsn, state: State) -> State:
	read, written = instruction.regs_access()
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
		escaped=state.escaped
		or (
			any(
				operand.type == CS_OP_REG
				and operand.reg in read
				and operand.reg in state.sp_offsets
				for operand in instruction.operands
			)
			and any(
				instruction.reg_name(register) not in FLAG_REGISTER_NAMES for register in written
			)
		),
		loaded_from=without_loaded(state.loaded_from, written),
	)


def stack_read(stack: Mapping[int, frozenset[Address]], offsets: OffsetSet) -> ValueSet:
	return bind(offsets, partial(_frame_read, stack))


def _frame_read(stack: Mapping[int, frozenset[Address]], offsets: frozenset[int]) -> ValueSet:
	if any(offset not in stack for offset in offsets):
		return Top()
	return capped(frozenset(value for offset in offsets for value in stack[offset]))


def frame_write(
	stack: Mapping[int, frozenset[Address]], offsets: OffsetSet, store: Store, pointer_size: int
) -> dict[int, frozenset[Address]]:
	match offsets:
		case Top():
			return {}
		case Known(values=starts):
			written = _written(
				store,
				pointer_size,
				starts,
				_overlapping(stack, store, pointer_size, starts)
				| _word_keys(store, pointer_size, starts),
			)
			return {
				**{key: values for key, values in stack.items() if key not in written},
				**dict(
					entry
					for key, value in written.items()
					for entry in map_entry(
						key, join(Known(values=stack[key]), value) if key in stack else value
					)
				),
			}
		case _ as unreachable:
			assert_never(unreachable)


def image_write(writes: Writes, addresses: ValueSet, store: Store, pointer_size: int) -> Writes:
	match addresses:
		case Top():
			return Writes(values={}, wild=True)
		case Known(values=starts):
			return Writes(
				values={
					**writes.values,
					**{
						Address(address): join(writes.values[Address(address)], value)
						if address in writes.values
						else value
						for address, value in _written(
							store,
							pointer_size,
							starts,
							_overlapping(writes.values, store, pointer_size, starts)
							| starts
							| _word_keys(store, pointer_size, starts)
							| _aligned_keys(store, pointer_size, starts),
						).items()
					},
				},
				wild=writes.wild,
			)
		case _ as unreachable:
			assert_never(unreachable)


def _written(
	store: Store, pointer_size: int, starts: frozenset[int], keys: Iterable[int]
) -> dict[int, ValueSet]:
	return {
		key: reduce(join, words)
		for key in keys
		if (
			words := tuple(
				_word(store, pointer_size, key - start)
				for start in starts
				if key < start + store.width and start < key + pointer_size
			)
		)
	}


def _word(store: Store, pointer_size: int, offset: int) -> ValueSet:
	index, remainder = divmod(offset, pointer_size)
	return (
		store.words[index]
		if remainder == 0
		and 0 <= index < len(store.words)
		and (index + 1) * pointer_size <= store.width
		else Top()
	)


def _overlapping(
	keys: Container[int], store: Store, pointer_size: int, starts: frozenset[int]
) -> frozenset[int]:
	return frozenset(
		key
		for start in starts
		for key in range(start - pointer_size + 1, start + store.width)
		if key in keys
	)


def _word_keys(store: Store, pointer_size: int, starts: frozenset[int]) -> frozenset[int]:
	return frozenset(
		start + index * pointer_size
		for start in starts
		for index in range(min(len(store.words), store.width // pointer_size))
	)


def _aligned_keys(store: Store, pointer_size: int, starts: frozenset[int]) -> frozenset[int]:
	return frozenset(
		key
		for start in starts
		for key in range(start - start % pointer_size, start + store.width, pointer_size)
	)


def copy_register(state: State, destination: int, source: int) -> State:
	if destination == source:
		return state
	if source in state.sp_offsets:
		return set_offsets(state, destination, Known(values=state.sp_offsets[source]))
	return set_loaded(
		state, destination, lookup(state.registers, source), state.loaded_from.get(source)
	)
