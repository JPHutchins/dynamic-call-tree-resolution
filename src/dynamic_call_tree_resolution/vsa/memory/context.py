# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""What the analysis knows of the image, and the writes the rounds accumulate."""

from __future__ import annotations

from typing import TYPE_CHECKING

from salix import Struct

from dynamic_call_tree_resolution.model import Address
from dynamic_call_tree_resolution.span import Span
from dynamic_call_tree_resolution.vsa.abi import (
	normalized,
)
from dynamic_call_tree_resolution.vsa.escape import NO_ESCAPES
from dynamic_call_tree_resolution.vsa.lattice import (
	join,
)
from dynamic_call_tree_resolution.vsa.state import (
	Writes,
)

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping

	from dynamic_call_tree_resolution.model import Program
	from dynamic_call_tree_resolution.vsa.entry_loads import EntryLoad
	from dynamic_call_tree_resolution.vsa.escape import Escapes
	from dynamic_call_tree_resolution.vsa.lattice import ValueSet


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
	entry_loads: Mapping[Address, tuple[EntryLoad, ...]] = {}
	"""Each function's entry loads, by where its code starts."""
	entry_words: Mapping[Address, ValueSet] = {}
	"""What every observed caller holds where an entry load reads, by the load's address."""
	upward_frame_writes: bool = False
	"""Whether a call or an unknown store writes a caller's frame only at or above the lowest
	frame address that escaped, so the analysis tracks that address."""


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
	entry_loads: Mapping[Address, tuple[EntryLoad, ...]] | None = None,
	entry_words: Mapping[Address, ValueSet] | None = None,
	upward_frame_writes: bool = False,
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
		entry_loads=entry_loads or {},
		entry_words=entry_words or {},
		upward_frame_writes=upward_frame_writes,
	)
