# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Reading the image and the global writes."""

from __future__ import annotations

from bisect import bisect_right
from functools import partial
from typing import TYPE_CHECKING, assert_never

from dynamic_call_tree_resolution.model import Address
from dynamic_call_tree_resolution.points_to import memory_at
from dynamic_call_tree_resolution.vsa.lattice import (
	K_BOUND,
	Known,
	Top,
	ValueSet,
	bind,
	capped,
	map_set,
	shift_addresses,
)
from dynamic_call_tree_resolution.vsa.state import (
	State,
	overlapping_keys,
	top_seed,
)

if TYPE_CHECKING:
	from dynamic_call_tree_resolution.span import Span
	from dynamic_call_tree_resolution.vsa.memory.context import Context


def _span_at(context: Context, address: Address) -> Span | None:
	index = bisect_right(context.object_starts, address) - 1
	if index < 0:
		return None
	span = context.object_spans[index]
	return span if span.start <= address < span.end else None


def _assumed_unwritten(context: Context, address: Address, width: int) -> bool:
	index = bisect_right(context.unwritten_starts, address) - 1
	return index >= 0 and address + width <= context.unwritten_spans[index].end


def kept(context: Context, clobbered: frozenset[Span], address: Address, width: int) -> bool:
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


def low_bytes(context: Context, values: ValueSet, width: int) -> ValueSet:
	"""The ``width`` bytes each word starts with, in the image's byte order."""
	return (
		values
		if width >= context.program.pointer_size
		else Top()
		if context.program.byte_order == "big"
		else map_set(values, lambda value: Address(value & ((1 << 8 * width) - 1)))
	)


def array_slots(context: Context, base: Address, start: Address, stride: int) -> ValueSet:
	"""The array's slots moved to start, when start lies in the section of the array's base."""
	index = bisect_right(context.section_starts, base) - 1
	return (
		shift_addresses(section_slots(context, base, stride), start - base)
		if index >= 0 and start in context.section_spans[index]
		else Top()
	)


def section_slots(context: Context, start: Address, stride: int) -> ValueSet:
	index = bisect_right(context.section_starts, start) - 1
	if index < 0 or stride <= 0 or _span_at(context, start) is None:
		return Top()
	count = (context.section_spans[index].end - start) // stride
	if not 0 < count <= K_BOUND:
		return Top()
	return Known(values=frozenset(Address(start + offset * stride) for offset in range(count)))


def image_value(context: Context, state: State, addresses: ValueSet, width: int) -> ValueSet:
	return bind(addresses, partial(image_read, context, state, width))


def runtime_value(context: Context, address: Address) -> ValueSet:
	return image_read(
		context,
		top_seed(context.program.machine),
		context.program.pointer_size,
		frozenset({address}),
	)


def image_read(
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
		match low_bytes(context, written, width) if written is not None else None:
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
			and not kept(
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
