# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Where a store writes, and what it leaves."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from capstone import x86_const
from salix import Struct, replace

from dynamic_call_tree_resolution.model import Address, InstructionFamily
from dynamic_call_tree_resolution.span import Span
from dynamic_call_tree_resolution.vsa.abi import (
	SP_REGISTERS,
)
from dynamic_call_tree_resolution.vsa.lattice import (
	Known,
	Lattice,
	OffsetSet,
	Top,
	ValueSet,
	join,
	join_maps,
	lookup,
	shift_addresses,
	shift_offsets,
)
from dynamic_call_tree_resolution.vsa.memory.addresses import (
	memory_addresses,
	stack_offsets,
)
from dynamic_call_tree_resolution.vsa.memory.reads import (
	image_read,
	kept,
)
from dynamic_call_tree_resolution.vsa.state import (
	State,
	Store,
	Writes,
	agreed_facts,
	frame_after_escape,
	frame_based,
	frame_write,
	image_write,
	join_escapes,
	lost_unless_anywhere,
)

if TYPE_CHECKING:
	from collections.abc import Hashable

	from capstone import CsInsn, CsOperand

	from dynamic_call_tree_resolution.vsa.memory.context import Context


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
	context: Context, state: State, instruction: CsInsn, operand: CsOperand
) -> Destination:
	machine = context.program.machine
	memory = operand.mem
	if frame_based(state, machine, memory.base):
		return Frame(offsets=stack_offsets(context, state, operand))
	if memory.base in SP_REGISTERS[machine] and memory.base not in state.registers:
		return Unknown()
	return Image(addresses=_store_addresses(context, state, instruction, operand))


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
		context,
		state,
		memory_destination(context, state, instruction, operand),
		store,
		Address(instruction.address),
	)


def store_at(
	context: Context, state: State, destination: Destination, store: Store, site: Address | None
) -> State:
	pointer_size = context.program.pointer_size
	match destination:
		case Frame(offsets=offsets):
			return _weak_across(
				context,
				state,
				offsets,
				replace(state, stack=frame_write(state.stack, offsets, store, pointer_size)),
			)
		case Image(addresses=addresses):
			return _image_store(context, state, addresses, store, site)
		case Unknown():
			return after_unknown_store(context, state)
		case _ as unreachable:
			assert_never(unreachable)


def _image_store(
	context: Context, state: State, addresses: ValueSet, store: Store, site: Address | None
) -> State:
	match addresses:
		case Top():
			return replace(
				state,
				stack=frame_after_escape(state),
				globals=_unknown_writes(context, state.globals, site),
			)
		case Known(values=starts):
			return _weak_across(
				context,
				state,
				addresses,
				replace(
					state,
					globals=image_write(state.globals, starts, store, context.program.pointer_size),
				),
			)
		case _ as unreachable:
			assert_never(unreachable)


def after_unknown_store(context: Context, state: State) -> State:
	"""The state after a store that may write the stack and any escaped object."""
	return replace(state, stack={}, globals=_unknown_writes(context, state.globals, None))


def _unknown_writes(context: Context, writes: Writes, site: Address | None) -> Writes:
	"""The writes after an unknown store, which may write any escaped object and its own targets."""
	clobbered = writes.clobbered | (
		context.store_targets.get(site, frozenset()) if site is not None else frozenset[Span]()
	)
	return Writes(
		values={
			address: value
			for address, value in writes.values.items()
			if kept(context, clobbered, address, context.program.pointer_size)
		},
		unknown_store=True,
		clobbered=clobbered,
	)


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
	sp_offsets = join_maps(before.sp_offsets, after.sp_offsets)
	return State(
		registers=join_maps(before.registers, after.registers),
		sp_offsets=sp_offsets,
		stack=join_maps(before.stack, after.stack),
		globals=Writes(
			values={
				address: value
				if before.globals.values.get(address) == value
				else join(
					value,
					image_read(context, before, context.program.pointer_size, frozenset({address})),
				)
				for address, value in after.globals.values.items()
			},
			unknown_store=after.globals.unknown_store,
			clobbered=before.globals.clobbered | after.globals.clobbered,
		),
		escaped=lost_unless_anywhere(
			join_escapes(before.escaped, after.escaped),
			before.sp_offsets,
			after.sp_offsets,
			sp_offsets,
		),
		loaded_from=agreed_facts(before.loaded_from, after.loaded_from),
		strided=agreed_facts(before.strided, after.strided),
	)


def _store_addresses(
	context: Context, state: State, instruction: CsInsn, operand: CsOperand
) -> ValueSet:
	memory = operand.mem
	match context.program.machine.family:
		case InstructionFamily.X86 if memory.base == x86_const.X86_REG_RIP:
			return Known(
				values=frozenset({Address(instruction.address + instruction.size + memory.disp)})
			)
		case InstructionFamily.X86 | InstructionFamily.ARM:
			return memory_addresses(context, state, operand)
		case _ as unreachable:
			assert_never(unreachable)
