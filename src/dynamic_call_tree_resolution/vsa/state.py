# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Abstract machine states and their transfer primitives."""

from __future__ import annotations

from functools import partial, reduce
from typing import TYPE_CHECKING, Final, assert_never

from capstone import CS_OP_REG
from salix import Struct, replace

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
	from dynamic_call_tree_resolution.span import Span


class Writes(Struct):
	"""What the analyzed stores wrote to program-global memory."""

	values: Mapping[Address, ValueSet]
	"""By address; an address no store wrote is absent."""
	unknown_store: bool
	"""An unknown store happened, so any escaped object may hold anything."""
	clobbered: frozenset[Span] = frozenset()
	"""The unescaped objects an unknown store derived from them may have written, by span."""


NO_WRITES: Final = Writes(values={}, unknown_store=False)


class State(Struct):
	"""Abstract state at one block entry; an absent entry is Top, an empty set Bottom."""

	registers: Mapping[int, frozenset[Address]]
	sp_offsets: Mapping[int, frozenset[int]]
	"""Copies of the stack pointer, as offsets from its entry value."""
	stack: Mapping[int, frozenset[Address]]
	"""Frame slots, by entry-frame offset."""
	globals: Writes
	"""What the stores on the path to the block wrote."""
	escaped: FrameEscape
	"""Whether a frame address may be held where the analysis does not track it."""
	loaded_from: Mapping[int, Address]
	"""The one image address each register's value was loaded from."""
	strided: Mapping[int, Strided]
	"""The registers holding an unknown index into an array, which an access resolves within the
	array's section."""


class Unescaped(Struct):
	"""No frame address is held where the analysis does not track it."""


class EscapedFrom(Struct):
	"""Frame addresses escaped, the lowest of them at an entry-frame offset."""

	lowest: int


class EscapedAnywhere(Struct):
	"""A frame address escaped where the analysis can't say which."""


type FrameEscape = Unescaped | EscapedFrom | EscapedAnywhere

UNESCAPED: Final = Unescaped()
ESCAPED_ANYWHERE: Final = EscapedAnywhere()


def join_escapes(current: FrameEscape, incoming: FrameEscape) -> FrameEscape:
	if current is incoming:
		return current
	match current, incoming:
		case Unescaped(), _:
			return incoming
		case _, Unescaped():
			return current
		case EscapedFrom(lowest=lowest), EscapedFrom(lowest=other):
			return current if lowest <= other else incoming
		case ((EscapedFrom() | EscapedAnywhere()), (EscapedFrom() | EscapedAnywhere())):
			return ESCAPED_ANYWHERE
		case _ as unreachable:
			assert_never(unreachable)


def frame_after_escape(state: State) -> Mapping[int, frozenset[Address]]:
	"""The frame slots a call or an unknown store leaves the analysis knowing."""
	match state.escaped:
		case Unescaped():
			return state.stack
		case EscapedFrom() | EscapedAnywhere():
			return {}
		case _ as unreachable:
			assert_never(unreachable)


class Strided(Struct):
	"""An array's base plus an unknown, non-negative index times a stride."""

	base: Address
	stride: int
	offset: int = 0
	"""The constant added to the indexed address since."""


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
		escaped=UNESCAPED,
		loaded_from={},
		strided={},
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
		escaped=lost_unless_anywhere(
			join_escapes(current.escaped, incoming.escaped),
			current.sp_offsets,
			incoming.sp_offsets,
			sp_offsets,
		),
		loaded_from=agreed_facts(current.loaded_from, incoming.loaded_from),
		strided=agreed_facts(current.strided, incoming.strided),
	)


def agreed_facts[T](current: Mapping[int, T], incoming: Mapping[int, T]) -> dict[int, T]:
	return {register: fact for register, fact in current.items() if incoming.get(register) == fact}


def lost_unless_anywhere(
	escaped: FrameEscape,
	current: Mapping[int, frozenset[int]],
	incoming: Mapping[int, frozenset[int]],
	joined: Mapping[int, frozenset[int]],
) -> FrameEscape:
	return (
		ESCAPED_ANYWHERE
		if not isinstance(escaped, EscapedAnywhere) and frame_copies_lost(current, incoming, joined)
		else escaped
	)


def frame_copies_lost(
	current: Mapping[int, frozenset[int]],
	incoming: Mapping[int, frozenset[int]],
	joined: Mapping[int, frozenset[int]],
) -> bool:
	"""Whether ``joined``, the ``join_maps`` of the two, dropped a register either one held.

	>>> frame_copies_lost({13: frozenset({0})}, {13: frozenset({-8})}, {13: frozenset({0, -8})})
	False
	>>> frame_copies_lost({13: frozenset({0}), 0: frozenset({-8})}, {13: frozenset({0})}, {13: frozenset({0})})
	True
	"""
	return len(joined) != len(current) or len(joined) != len(incoming)


def _join_writes(current: Writes, incoming: Writes) -> Writes:
	return Writes(
		values={
			address: join(current.values[address], incoming.values[address])
			for address in current.values.keys() & incoming.values.keys()
		},
		unknown_store=current.unknown_store or incoming.unknown_store,
		clobbered=current.clobbered | incoming.clobbered,
	)


def set_register(state: State, register: int, value: ValueSet) -> State:
	return set_loaded(state, register, value, None)


def set_strided(state: State, register: int, strided: Strided) -> State:
	return set_loaded(state, register, Top(), None, strided)


def set_loaded(
	state: State,
	register: int,
	value: ValueSet,
	source: Address | None,
	strided: Strided | None = None,
) -> State:
	sp_offsets = dict(state.sp_offsets)
	sp_offsets.pop(register, None)
	return State(
		registers=put_value(state.registers, register, value),
		sp_offsets=sp_offsets,
		stack=state.stack,
		globals=state.globals,
		escaped=state.escaped,
		loaded_from=_with_fact(state.loaded_from, register, source),
		strided=_with_fact(state.strided, register, strided),
	)


def _with_fact[T](facts: Mapping[int, T], register: int, fact: T | None) -> Mapping[int, T]:
	return (
		{**facts, register: fact}
		if fact is not None
		else {held: kept for held, kept in facts.items() if held != register}
		if register in facts
		else facts
	)


def without_facts[T](facts: Mapping[int, T], registers: Collection[int]) -> Mapping[int, T]:
	return (
		facts
		if facts.keys().isdisjoint(registers)
		else {register: fact for register, fact in facts.items() if register not in registers}
	)


def set_offsets(state: State, register: int, value: OffsetSet) -> State:
	registers = dict(state.registers)
	registers.pop(register, None)
	return State(
		registers=registers,
		sp_offsets=put_value(state.sp_offsets, register, value),
		stack=state.stack,
		globals=state.globals,
		escaped=ESCAPED_ANYWHERE if is_top(value) else state.escaped,
		loaded_from=_with_fact(state.loaded_from, register, None),
		strided=_with_fact(state.strided, register, None),
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
		loaded_from=without_facts(state.loaded_from, registers),
		strided=without_facts(state.strided, registers),
	)


def frame_based(state: State, machine: Machine, register: int) -> bool:
	return register == SP_REGISTERS[machine][0] or register in state.sp_offsets


def escaping(state: State, registers: Iterable[int]) -> State:
	frame_registers = [register for register in registers if register in state.sp_offsets]
	if not frame_registers:
		return state
	lowest = min(
		(offset for register in frame_registers for offset in state.sp_offsets[register]),
		default=None,
	)
	if lowest is None:
		return state
	match state.escaped:
		case EscapedAnywhere():
			return state
		case EscapedFrom(lowest=escaped) if escaped <= lowest:
			return state
		case Unescaped() | EscapedFrom():
			return replace(state, escaped=EscapedFrom(lowest))
		case _ as unreachable:
			assert_never(unreachable)


def top_written(instruction: CsInsn, state: State, roots: Mapping[int, int] | None = None) -> State:
	read, written = instruction.regs_access()
	dropped = (
		written
		if roots is None
		else {*written, *(roots.get(register, register) for register in written)}
	)
	return State(
		registers={
			register: value
			for register, value in state.registers.items()
			if register not in dropped
		},
		sp_offsets={
			register: value
			for register, value in state.sp_offsets.items()
			if register not in dropped
		},
		stack=state.stack,
		globals=state.globals,
		escaped=ESCAPED_ANYWHERE
		if isinstance(state.escaped, EscapedAnywhere)
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
		)
		else state.escaped,
		loaded_from=without_facts(state.loaded_from, dropped),
		strided=without_facts(state.strided, dropped),
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
				**{
					entry: values
					for key, value in written.items()
					for entry, values in map_entry(key, value).items()
				},
			}
		case _ as unreachable:
			assert_never(unreachable)


def image_write(
	writes: Writes, starts: frozenset[Address], store: Store, pointer_size: int
) -> Writes:
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
					| _word_keys(store, pointer_size, starts)
					| _aligned_keys(store, pointer_size, starts),
				).items()
			},
		},
		unknown_store=writes.unknown_store,
		clobbered=writes.clobbered,
	)


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
		key for start in starts for key in overlapping_keys(keys, start, store.width, pointer_size)
	)


def overlapping_keys(
	keys: Container[int], start: int, width: int, pointer_size: int
) -> frozenset[int]:
	return frozenset(key for key in range(start - pointer_size + 1, start + width) if key in keys)


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
		state,
		destination,
		lookup(state.registers, source),
		state.loaded_from.get(source),
		state.strided.get(source),
	)
