# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The writable objects of an image, and the objects an address points into."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from enum import Enum
from itertools import accumulate, chain, pairwise
from typing import TYPE_CHECKING, Final, assert_never

from salix import Struct

from dynamic_call_tree_resolution.model import Address
from dynamic_call_tree_resolution.points_to import in_writable_memory, memory_at
from dynamic_call_tree_resolution.span import Span
from dynamic_call_tree_resolution.vsa.abi import (
	is_data,
	normalized,
)
from dynamic_call_tree_resolution.vsa.cfg import branch_target

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping

	from dynamic_call_tree_resolution.model import Program
	from dynamic_call_tree_resolution.vsa.cfg import Block, FunctionGraph
	from dynamic_call_tree_resolution.vsa.escape.values import Held


_NEARBY: Final = 0x1000


class Fill(Enum):
	"""What a fill routine writes over its destination."""

	COPY = "copy"
	ZERO = "zero"


_FILL_ROUTINES: Final = {"memcpy": Fill.COPY, "memset": Fill.ZERO}


class Image(Struct):
	"""What attributing an address needs to know about the image, computed once."""

	program: Program
	spans: tuple[Span, ...]
	"""Writable objects, with overlapping objects merged, sorted."""
	starts: tuple[Address, ...]
	nearby: tuple[int, int]
	"""The addresses close enough to a writable object to follow as possible addresses."""
	bounded: Mapping[int, tuple[Span, ...]]
	"""The writable sections each linker label starts or lies in; one that ends at a label is
	past it, as an object is past a pointer to its end."""
	walkable: Mapping[int, tuple[Span, ...]]
	"""Of those, the ones code may walk as an array from the label: every one at a label that
	starts no object, else those whose objects share one type, as an iterable section's do."""
	fills: Mapping[Address, Fill]
	"""``memcpy``, ``memset``, and functions whose only instruction branches to one of them."""
	parameterless: frozenset[Address]
	"""Functions that read no argument registers."""


def escape_image(program: Program, graphs: tuple[FunctionGraph, ...]) -> Image:
	spans = _merged(
		Span(start=data_object.address, end=Address(data_object.address + data_object.size))
		for data_object in program.objects.values()
		if data_object.size > 0 and in_writable_memory(program, data_object.address)
	)
	writable = tuple(
		sorted(
			(
				*(
					Span(start=start, end=Address(start + len(section.data)))
					for start, section in program.sections.items()
					if section.writable and section.data
				),
				*program.nobits,
			)
		)
	)
	bounds = (*spans, *writable)
	bounded: Mapping[int, tuple[Span, ...]] = {
		address: sections
		for address in program.labels.values()
		if (sections := tuple(span for span in writable if span.start <= address < span.end))
	}
	object_starts = frozenset(data_object.address for data_object in program.objects.values())
	arrays = frozenset(section for section in writable if _one_type(program, section))
	routines = {
		normalized(address, program.machine): fill
		for name, fill in _FILL_ROUTINES.items()
		for address in program.symbol_addresses.get(name, frozenset())
	}
	return Image(
		program=program,
		spans=spans,
		starts=tuple(span.start for span in spans),
		nearby=(
			min(span.start for span in bounds) - _NEARBY,
			max(span.end for span in bounds) + _NEARBY,
		)
		if bounds
		else (0, -1),
		bounded=bounded,
		walkable={
			address: walkable
			for address, sections in bounded.items()
			if (
				walkable := tuple(
					section
					for section in sections
					if address not in object_starts or section in arrays
				)
			)
		},
		parameterless=frozenset(
			normalized(address, program.machine) for address in program.parameterless
		),
		fills={
			**routines,
			**{
				normalized(graph.function.address, program.machine): fill
				for graph in graphs
				if (fill := _forwarded_fill(program, routines, graph.blocks)) is not None
			},
		},
	)


def _one_type(program: Program, section: Span) -> bool:
	return (
		len(
			{
				data_object.type_name
				for data_object in program.objects.values()
				if data_object.type_name is not None
				and section.start <= data_object.address < section.end
			}
		)
		<= 1
	)


def _merged(spans: Iterable[Span]) -> tuple[Span, ...]:
	ordered = sorted(spans)
	reach = tuple(accumulate((span.end for span in ordered), max))
	breaks = tuple(
		index
		for index in range(len(ordered))
		if index == 0 or ordered[index].start >= reach[index - 1]
	)
	return tuple(
		Span(start=ordered[first].start, end=reach[after - 1])
		for first, after in pairwise((*breaks, len(ordered)))
	)


def _forwarded_fill(
	program: Program, routines: Mapping[Address, Fill], blocks: tuple[Block, ...]
) -> Fill | None:
	match tuple(
		instruction
		for block in blocks
		for instruction in block.instructions
		if not is_data(instruction)
	):
		case (only,):
			branch = branch_target(only, program.machine)
			return (
				routines.get(normalized(Address(branch.target), program.machine))
				if branch is not None and not branch.conditional
				else None
			)
		case _:
			return None


def reproduced(image: Image, fill: Fill, span: Span, destination: int, argument: int) -> bool:
	match fill:
		case Fill.COPY:
			return any(
				section.writable
				and section.load_address is not None
				and start <= span.start
				and span.end <= start + len(section.data)
				and section.load_address - start == argument - destination
				for start, section in image.program.sections.items()
			)
		case Fill.ZERO:
			return argument == 0 and any(
				nobits.start <= span.start and span.end <= nobits.end
				for nobits in image.program.nobits
			)
		case _ as unreachable:
			assert_never(unreachable)


def reached_objects(image: Image, held: Held) -> frozenset[Span]:
	"""The objects the held addresses point into, under C's pointer provenance."""
	return frozenset(
		chain(
			chain.from_iterable(_attributed(image, derived.address) for derived in held.exact),
			chain.from_iterable(
				containing(image, derived.address)
				| sections_objects(image, image.walkable, derived.address)
				| sections_objects(image, image.walkable, derived.origin)
				for derived in held.indexed
			),
		)
	)


def _attributed(image: Image, address: int) -> frozenset[Span]:
	"""The object an address points into, and the objects code may walk from a label there."""
	return containing(image, address) | sections_objects(image, image.walkable, address)


def containing(image: Image, address: int) -> frozenset[Span]:
	index = bisect_right(image.starts, address) - 1
	return (
		frozenset({image.spans[index]})
		if index >= 0 and address < image.spans[index].end
		else frozenset()
	)


def overlapping(image: Image, low: int, high: int) -> frozenset[Span]:
	return frozenset(
		span
		for span in image.spans[
			max(bisect_right(image.starts, low) - 1, 0) : bisect_left(image.starts, high)
		]
		if span.end > low
	)


def sections_objects(
	image: Image, sections: Mapping[int, tuple[Span, ...]], address: int
) -> frozenset[Span]:
	return frozenset(
		chain.from_iterable(
			overlapping(image, section.start, section.end) for section in sections.get(address, ())
		)
	)


def data_escapes(image: Image, literals: frozenset[int]) -> frozenset[Span]:
	"""The objects a word of data at any byte offset points into, outside code and its literals."""
	program = image.program
	return frozenset(
		chain.from_iterable(
			_attributed(image, word)
			for address, word in chain(
				chain.from_iterable(
					_words(program, start, section.data)
					for start, section in program.sections.items()
					if not section.executable
				),
				chain.from_iterable(
					_words(
						program, span.start, memory_at(program, span.start, span.end - span.start)
					)
					for span in program.data_in_code
				),
			)
			if image.nearby[0] <= word <= image.nearby[1] and address not in literals
		)
	)


def _words(program: Program, start: int, data: bytes) -> Iterable[tuple[int, int]]:
	"""Each word at each byte offset of the data that starts at the address, with its address."""
	return (
		(start + offset, int.from_bytes(data[offset : offset + 4], program.byte_order))
		for offset in range(len(data) - 3)
	)
