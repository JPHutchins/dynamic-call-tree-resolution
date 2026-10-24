# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Loads and stores against the loaded image and the global writes."""

from __future__ import annotations

from bisect import bisect_right
from functools import partial
from typing import TYPE_CHECKING, assert_never

from capstone import arm_const, x86_const
from salix import Struct, replace

from dynamic_call_tree_resolution.model import Address, InstructionFamily
from dynamic_call_tree_resolution.points_to import instruction_set_at, memory_at
from dynamic_call_tree_resolution.span import Span
from dynamic_call_tree_resolution.vsa.abi import (
	SP_REGISTERS,
	index_scale,
	normalized,
	program_counter,
)
from dynamic_call_tree_resolution.vsa.escape import NO_ESCAPES
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
	map_set,
	shift_addresses,
	shift_offsets,
)
from dynamic_call_tree_resolution.vsa.state import (
	State,
	Store,
	Strided,
	Writes,
	agreed_facts,
	frame_based,
	frame_copies_lost,
	frame_write,
	image_write,
	overlapping_keys,
	stack_read,
	top_seed,
)

if TYPE_CHECKING:
	from collections.abc import Hashable, Iterable, Mapping

	from capstone import CsInsn, CsMemOperand, CsOperand

	from dynamic_call_tree_resolution.model import Machine, Program
	from dynamic_call_tree_resolution.vsa.escape import Escapes


class Context(Struct):
	"""What every function's interpretation reads from the program."""

	program: Program
	object_spans: tuple[Span, ...]
	object_starts: tuple[int, ...]
	section_spans: tuple[Span, ...]
	section_starts: tuple[int, ...]
	read_only_spans: tuple[Span, ...]
	read_only_starts: tuple[int, ...]
	global_writes: Writes
	global_covered: frozenset[int]
	"""Every byte a word of the global writes covers."""
	function_starts: frozenset[Address]
	external_symbols: Mapping[Address, str]
	"""The undefined symbol whose address the dynamic loader writes to each slot."""
	unwritten_spans: tuple[Span, ...] = ()
	"""The objects the user asserts the program never writes, by address span."""
	unwritten_starts: tuple[int, ...] = ()
	unescaped_spans: tuple[Span, ...] = ()
	"""The objects only the code that names them can reach, which an unknown store writes only
	through an address derived from them."""
	unescaped_starts: tuple[int, ...] = ()
	store_targets: Mapping[Address, frozenset[Span]] = {}
	"""The unescaped objects each store, by address, may write when the analysis can't place it."""


def accumulate_writes(current: Writes, written: Iterable[Writes]) -> Writes:
	values = dict(current.values)
	unknown_store = current.unknown_store
	clobbered = current.clobbered
	for writes in written:
		unknown_store = unknown_store or writes.unknown_store
		clobbered = clobbered | writes.clobbered
		for address, value in writes.values.items():
			values[address] = join(values[address], value) if address in values else value
	return Writes(values=values, unknown_store=unknown_store, clobbered=clobbered)


def context_for(
	program: Program,
	global_writes: Writes,
	*,
	assumed_unwritten: tuple[Span, ...] = (),
	escapes: Escapes = NO_ESCAPES,
) -> Context:
	spans = tuple(
		sorted(
			Span(start=object_.address, end=Address(object_.address + object_.size))
			for object_ in program.objects.values()
			if object_.size > 0
		)
	)
	sections = tuple(
		Span(start=start, end=Address(start + len(section.data)))
		for start, section in sorted(program.sections.items())
	)
	read_only = tuple(span for span in sections if not program.sections[span.start].writable)
	return Context(
		program=program,
		object_spans=spans,
		object_starts=tuple(span.start for span in spans),
		section_spans=sections,
		section_starts=tuple(span.start for span in sections),
		read_only_spans=read_only,
		read_only_starts=tuple(span.start for span in read_only),
		global_writes=global_writes,
		global_covered=frozenset(
			key + offset for key in global_writes.values for offset in range(program.pointer_size)
		),
		function_starts=frozenset(
			normalized(function.address, program.machine) for function in program.functions.values()
		),
		external_symbols={
			relocation.slot: relocation.symbol
			for relocation in program.relocations
			if relocation.symbol is not None
		},
		unwritten_spans=tuple(sorted(assumed_unwritten)),
		unwritten_starts=tuple(span.start for span in sorted(assumed_unwritten)),
		unescaped_spans=escapes.unescaped,
		unescaped_starts=tuple(span.start for span in escapes.unescaped),
		store_targets=escapes.store_targets,
	)


def _span_at(context: Context, address: Address) -> Span | None:
	index = bisect_right(context.object_starts, address) - 1
	if index < 0:
		return None
	span = context.object_spans[index]
	return span if span.start <= address < span.end else None


def _assumed_unwritten(context: Context, address: Address, width: int) -> bool:
	index = bisect_right(context.unwritten_starts, address) - 1
	return index >= 0 and address + width <= context.unwritten_spans[index].end


def _kept(context: Context, clobbered: frozenset[Span], address: Address, width: int) -> bool:
	"""Whether the bytes lie in one unescaped object no unknown store derived from it has written."""
	index = bisect_right(context.unescaped_starts, address) - 1
	return (
		index >= 0
		and address + width <= context.unescaped_spans[index].end
		and context.unescaped_spans[index] not in clobbered
	)


def _read_only(context: Context, address: Address) -> bool:
	index = bisect_right(context.read_only_starts, address) - 1
	return index >= 0 and address < context.read_only_spans[index].end


def _initial_value(context: Context, address: Address, width: int) -> Address | None:
	span = _span_at(context, address)
	data = memory_at(context.program, address, width)
	return (
		None
		if (span is not None and address + width > span.end) or len(data) != width
		else Address(int.from_bytes(data, context.program.byte_order))
	)


def _low_bytes(context: Context, values: ValueSet, width: int) -> ValueSet:
	"""The ``width`` bytes each word starts with, in the image's byte order."""
	return (
		values
		if width >= context.program.pointer_size
		else Top()
		if context.program.byte_order == "big"
		else map_set(values, lambda value: Address(value & ((1 << 8 * width) - 1)))
	)


def _array_slots(context: Context, base: Address, start: Address, stride: int) -> ValueSet:
	"""The array's slots moved to start, when start lies in the section of the array's base."""
	index = bisect_right(context.section_starts, base) - 1
	return (
		shift_addresses(_section_slots(context, base, stride), start - base)
		if index >= 0 and start in context.section_spans[index]
		else Top()
	)


def _section_slots(context: Context, start: Address, stride: int) -> ValueSet:
	index = bisect_right(context.section_starts, start) - 1
	if index < 0 or stride <= 0 or _span_at(context, start) is None:
		return Top()
	count = (context.section_spans[index].end - start) // stride
	if not 0 < count <= K_BOUND:
		return Top()
	return Known(values=frozenset(Address(start + offset * stride) for offset in range(count)))


def _image_value(context: Context, state: State, addresses: ValueSet, width: int) -> ValueSet:
	return bind(addresses, partial(_image_read, context, state, width))


def runtime_value(context: Context, address: Address) -> ValueSet:
	return _image_read(
		context,
		top_seed(context.program.machine),
		context.program.pointer_size,
		frozenset({address}),
	)


def _image_read(
	context: Context, state: State, width: int, addresses: frozenset[Address]
) -> ValueSet:
	values: set[Address] = set()
	for address in addresses:
		if context.unwritten_starts and _assumed_unwritten(context, address, width):
			initial = _initial_value(context, address, width)
			if initial is None:
				return Top()
			values.add(initial)
			continue
		if _overlapped(context, state, address, width):
			return Top()
		local = state.globals.values.get(address)
		written = local if local is not None else context.global_writes.values.get(address)
		match _low_bytes(context, written, width) if written is not None else None:
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
			and (state.globals.unknown_store or context.global_writes.unknown_store)
			and not _read_only(context, address)
			and not _kept(
				context,
				state.globals.clobbered | context.global_writes.clobbered,
				address,
				width,
			)
		) or address in context.external_symbols:
			return Top()
		value = _initial_value(context, address, width)
		if value is not None:
			values.add(value)
		elif written is None and _span_at(context, address) is not None:
			return Top()
	return capped(frozenset(values))


def _overlapped(context: Context, state: State, address: Address, width: int) -> bool:
	"""Whether a write recorded at another address covers part of the bytes read here."""
	pointer_size = context.program.pointer_size
	return (
		bool(state.globals.values)
		and bool(overlapping_keys(state.globals.values, address, width, pointer_size) - {address})
	) or (
		not context.global_covered.isdisjoint(range(address, address + width))
		and bool(
			overlapping_keys(context.global_writes.values, address, width, pointer_size) - {address}
		)
	)


def stack_offsets(context: Context, state: State, operand: CsOperand) -> OffsetSet:
	memory = operand.mem
	offsets = lookup(state.sp_offsets, memory.base)
	if memory.index == 0:
		return shift_offsets(offsets, memory.disp)
	match index_scale(context.program.machine, operand):
		case None:
			return Top()
		case int() as scale:
			return bind(
				offsets,
				lambda bases: bind(
					lookup(state.registers, memory.index),
					lambda indexes: indexed_offsets(bases, indexes, scale, memory.disp),
				),
			)
		case _ as unreachable:
			assert_never(unreachable)


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


def memory_addresses(context: Context, state: State, operand: CsOperand) -> ValueSet:
	memory = operand.mem
	base = (
		Known(values=frozenset({Address(0)}))
		if memory.base == 0
		else lookup(state.registers, memory.base)
	)
	if memory.index == 0:
		return _displaced(context, state.strided.get(memory.base), base, memory.disp)
	match index_scale(context.program.machine, operand):
		case None:
			return Top()
		case int() as scale:
			return scaled_addresses(
				context, base, lookup(state.registers, memory.index), scale, memory.disp
			)
		case _ as unreachable:
			assert_never(unreachable)


def _displaced(context: Context, strided: Strided | None, base: ValueSet, disp: int) -> ValueSet:
	match strided:
		case None:
			return shift_addresses(base, disp)
		case Strided(base=array, stride=stride, offset=offset):
			return _array_slots(context, array, Address(array + offset + disp), stride)
		case _ as unreachable:
			assert_never(unreachable)


class Loaded(Struct):
	"""A load's value, zero-extended from its width, and the one image address it reads."""

	value: ValueSet
	origin: Address | None


def load_value(context: Context, state: State, instruction: CsInsn, operand: CsOperand) -> ValueSet:
	return loaded(context, state, instruction, operand, context.program.pointer_size).value


def loaded(
	context: Context, state: State, instruction: CsInsn, operand: CsOperand, width: int
) -> Loaded:
	match _load_addresses(context, state, instruction, operand):
		case None:
			return Loaded(
				_low_bytes(
					context, stack_read(state.stack, stack_offsets(context, state, operand)), width
				),
				None,
			)
		case Known(values=values) as addresses if len(values) == 1:
			return Loaded(_image_value(context, state, addresses, width), next(iter(values)))
		case Known() | Top() as addresses:
			return Loaded(
				_image_value(context, state, addresses, width)
				if width >= context.program.pointer_size
				else Top(),
				None,
			)
		case _ as unreachable:
			assert_never(unreachable)


def _load_addresses(
	context: Context, state: State, instruction: CsInsn, operand: CsOperand
) -> ValueSet | None:
	memory = operand.mem
	if _pc_relative(context.program.machine, memory):
		return Known(
			values=frozenset(
				{
					Address(
						program_counter(
							instruction,
							instruction_set_at(context.program, Address(instruction.address)),
						)
						+ memory.disp
					)
				}
			)
		)
	if frame_based(state, context.program.machine, memory.base):
		return None
	return memory_addresses(context, state, operand)


def _pc_relative(machine: Machine, memory: CsMemOperand) -> bool:
	match machine.family:
		case InstructionFamily.X86:
			return memory.base == x86_const.X86_REG_RIP
		case InstructionFamily.ARM:
			return memory.base == arm_const.ARM_REG_PC and memory.index == 0
		case _ as unreachable:
			assert_never(unreachable)


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
				stack={} if state.escaped else state.stack,
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
			if _kept(context, clobbered, address, context.program.pointer_size)
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
					_image_read(
						context, before, context.program.pointer_size, frozenset({address})
					),
				)
				for address, value in after.globals.values.items()
			},
			unknown_store=after.globals.unknown_store,
			clobbered=before.globals.clobbered | after.globals.clobbered,
		),
		escaped=before.escaped
		or after.escaped
		or frame_copies_lost(before.sp_offsets, after.sp_offsets, sp_offsets),
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
