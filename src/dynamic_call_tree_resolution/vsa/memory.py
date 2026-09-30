# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Loads and stores against the loaded image and the global writes."""

from __future__ import annotations

from bisect import bisect_right
from functools import partial
from typing import TYPE_CHECKING, assert_never

from capstone import arm_const, x86_const
from salix import Struct

from dynamic_call_tree_resolution.model import Address
from dynamic_call_tree_resolution.points_to import read_pointer
from dynamic_call_tree_resolution.vsa.abi import SP_REGISTERS, normalized
from dynamic_call_tree_resolution.vsa.lattice import (
	K_BOUND,
	Known,
	Lattice,
	OffsetSet,
	Top,
	ValueSet,
	bind,
	capped,
	indexed,
	indexed_offsets,
	join,
	join_maps,
	lookup,
	shift_addresses,
	shift_offsets,
)
from dynamic_call_tree_resolution.vsa.state import (
	State,
	Store,
	Writes,
	frame_based,
	frame_write,
	image_write,
	stack_read,
	unknown_memory,
)

if TYPE_CHECKING:
	from collections.abc import Hashable, Iterable

	from capstone import CsInsn, CsMemOperand, CsOperand

	from dynamic_call_tree_resolution.model import Program


class Context(Struct):
	"""What every function's interpretation reads from the program."""

	program: Program
	object_spans: tuple[tuple[int, int], ...]
	object_starts: tuple[int, ...]
	section_spans: tuple[tuple[int, int], ...]
	section_starts: tuple[int, ...]
	read_only_spans: tuple[tuple[int, int], ...]
	read_only_starts: tuple[int, ...]
	global_writes: Writes
	function_starts: frozenset[Address]


def accumulate_writes(current: Writes, written: Iterable[Writes]) -> Writes:
	values = dict(current.values)
	wild = current.wild
	for writes in written:
		wild = wild or writes.wild
		for address, value in writes.values.items():
			values[address] = join(values[address], value) if address in values else value
	return Writes(values=values, wild=wild)


def context_for(program: Program, global_writes: Writes) -> Context:
	spans = tuple(
		sorted(
			(object_.address, object_.address + object_.size)
			for object_ in program.objects.values()
			if object_.size > 0
		)
	)
	sections = tuple(
		sorted(
			(start, start + len(section.data), section.writable)
			for start, section in program.sections.items()
		)
	)
	read_only = tuple((start, end) for start, end, writable in sections if not writable)
	return Context(
		program=program,
		object_spans=spans,
		object_starts=tuple(span[0] for span in spans),
		section_spans=tuple((start, end) for start, end, _ in sections),
		section_starts=tuple(start for start, _, _ in sections),
		read_only_spans=read_only,
		read_only_starts=tuple(span[0] for span in read_only),
		global_writes=global_writes,
		function_starts=frozenset(
			normalized(function.address, program.machine) for function in program.functions.values()
		),
	)


def _span_at(context: Context, address: Address) -> tuple[int, int] | None:
	index = bisect_right(context.object_starts, address) - 1
	if index < 0:
		return None
	span = context.object_spans[index]
	return span if span[0] <= address < span[1] else None


def _read_only(context: Context, address: Address) -> bool:
	index = bisect_right(context.read_only_starts, address) - 1
	return index >= 0 and address < context.read_only_spans[index][1]


def _pointer_value(context: Context, address: Address) -> Address | None:
	span = _span_at(context, address)
	return read_pointer(context.program, address, span[1] if span is not None else None)


def _section_slots(context: Context, start: Address, stride: int) -> ValueSet:
	index = bisect_right(context.section_starts, start) - 1
	if index < 0 or stride <= 0 or _span_at(context, start) is None:
		return Top()
	count = (context.section_spans[index][1] - start) // stride
	if not 0 < count <= K_BOUND:
		return Top()
	return Known(values=frozenset(Address(start + offset * stride) for offset in range(count)))


def _image_value(context: Context, state: State, addresses: ValueSet) -> ValueSet:
	return bind(addresses, partial(_image_read, context, state))


def _image_read(context: Context, state: State, addresses: frozenset[Address]) -> ValueSet:
	values: set[Address] = set()
	for address in addresses:
		local = state.globals.values.get(address)
		written = local if local is not None else context.global_writes.values.get(address)
		match written:
			case Top():
				return Top()
			case Known(values=known):
				values.update(known)
			case None:
				pass
			case _ as unreachable:
				assert_never(unreachable)
		if (
			local is None
			and (state.globals.wild or context.global_writes.wild)
			and not _read_only(context, address)
		):
			return Top()
		value = _pointer_value(context, address)
		if value is not None:
			values.add(value)
		elif written is None and _span_at(context, address) is not None:
			return Top()
	return capped(frozenset(values))


def stack_offsets(state: State, memory: CsMemOperand, base_register: int) -> OffsetSet:
	offsets = lookup(state.sp_offsets, base_register)
	if memory.index == 0:
		return shift_offsets(offsets, memory.disp)
	return bind(
		offsets,
		lambda bases: bind(
			lookup(state.registers, memory.index),
			lambda indexes: indexed_offsets(bases, indexes, memory.scale, memory.disp),
		),
	)


def scaled_addresses(
	context: Context, base: ValueSet, index: ValueSet, scale: int, disp: int
) -> ValueSet:
	match index:
		case Top():
			return bind(
				base,
				lambda bases: (
					_section_slots(context, Address(next(iter(bases)) + disp), scale)
					if len(bases) == 1
					else Top()
				),
			)
		case Known():
			return indexed(base, index, scale, disp)
		case _ as unreachable:
			assert_never(unreachable)


def memory_addresses(context: Context, state: State, memory: CsMemOperand) -> ValueSet:
	base = (
		Known(values=frozenset({Address(0)}))
		if memory.base == 0
		else lookup(state.registers, memory.base)
	)
	if memory.index == 0:
		return shift_addresses(base, memory.disp)
	return scaled_addresses(
		context, base, lookup(state.registers, memory.index), memory.scale, memory.disp
	)


def load_value(context: Context, state: State, instruction: CsInsn, operand: CsOperand) -> ValueSet:
	machine = context.program.machine
	memory = operand.mem
	if machine.is_x86:
		if memory.base == x86_const.X86_REG_RIP:
			address = Address(instruction.address + instruction.size + memory.disp)
			return _image_value(context, state, Known(values=frozenset({address})))
		if frame_based(state, machine, memory.base):
			return stack_read(state.stack, stack_offsets(state, memory, memory.base))
		return _image_value(context, state, memory_addresses(context, state, memory))
	if memory.base == arm_const.ARM_REG_PC:
		address = Address(((instruction.address + 4) & ~3) + memory.disp)
		return _image_value(context, state, Known(values=frozenset({address})))
	if memory.base == arm_const.ARM_REG_SP or memory.base in state.sp_offsets:
		return stack_read(state.stack, stack_offsets(state, memory, memory.base))
	return _image_value(context, state, memory_addresses(context, state, memory))


class Frame(Struct):
	"""Stack memory, by entry-frame offset."""

	offsets: OffsetSet


class Image(Struct):
	"""Program-global memory."""

	addresses: ValueSet


class Unknown(Struct):
	"""Anywhere: the frame or program-global memory."""


type Destination = Frame | Image | Unknown


def memory_destination(
	context: Context, state: State, instruction: CsInsn, memory: CsMemOperand
) -> Destination:
	machine = context.program.machine
	if frame_based(state, machine, memory.base):
		return Frame(offsets=stack_offsets(state, memory, memory.base))
	if memory.base in SP_REGISTERS[machine] and memory.base not in state.registers:
		return Unknown()
	return Image(addresses=_store_addresses(context, state, instruction, memory))


def register_destination(context: Context, state: State, register: int, delta: int) -> Destination:
	if frame_based(state, context.program.machine, register):
		return Frame(offsets=shift_offsets(lookup(state.sp_offsets, register), delta))
	return Image(addresses=shift_addresses(lookup(state.registers, register), delta))


def anywhere(destination: Destination) -> Destination:
	match destination:
		case Frame():
			return Frame(offsets=Top())
		case Image():
			return Image(addresses=Top())
		case Unknown():
			return destination
		case _ as unreachable:
			assert_never(unreachable)


def store_value(
	context: Context, state: State, instruction: CsInsn, operand: CsOperand, store: Store
) -> State:
	return store_at(
		context, state, memory_destination(context, state, instruction, operand.mem), store
	)


def store_at(context: Context, state: State, destination: Destination, store: Store) -> State:
	pointer_size = context.program.pointer_size
	match destination:
		case Frame(offsets=offsets):
			return _weak_across(
				context,
				state,
				offsets,
				State(
					registers=state.registers,
					sp_offsets=state.sp_offsets,
					stack=frame_write(state.stack, offsets, store, pointer_size),
					globals=state.globals,
				),
			)
		case Image(addresses=addresses):
			return _weak_across(
				context,
				state,
				addresses,
				State(
					registers=state.registers,
					sp_offsets=state.sp_offsets,
					stack=state.stack,
					globals=image_write(state.globals, addresses, store, pointer_size),
				),
			)
		case Unknown():
			return unknown_memory(state)
		case _ as unreachable:
			assert_never(unreachable)


def _weak_across[T: Hashable](
	context: Context, before: State, targets: Lattice[T], after: State
) -> State:
	match targets:
		case Top():
			return after
		case Known(values=values):
			return weakened(context, before, after) if len(values) > 1 else after
		case _ as unreachable:
			assert_never(unreachable)


def weakened(context: Context, before: State, after: State) -> State:
	return State(
		registers=join_maps(before.registers, after.registers),
		sp_offsets=join_maps(before.sp_offsets, after.sp_offsets),
		stack=join_maps(before.stack, after.stack),
		globals=Writes(
			values={
				address: value
				if before.globals.values.get(address) == value
				else join(value, _image_read(context, before, frozenset({address})))
				for address, value in after.globals.values.items()
			},
			wild=after.globals.wild,
		),
	)


def _store_addresses(
	context: Context, state: State, instruction: CsInsn, memory: CsMemOperand
) -> ValueSet:
	machine = context.program.machine
	if machine.is_x86 and memory.base == x86_const.X86_REG_RIP:
		return Known(
			values=frozenset({Address(instruction.address + instruction.size + memory.disp)})
		)
	return memory_addresses(context, state, memory)
