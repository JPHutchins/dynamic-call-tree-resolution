# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Source lines and inlined copies, as address spans."""

from __future__ import annotations

from functools import partial
from itertools import accumulate, groupby, pairwise
from typing import TYPE_CHECKING, assert_never

from elftools.dwarf.die import AttributeValue
from elftools.dwarf.ranges import BaseAddressEntry, RangeEntry
from elftools.elf.elffile import ELFFile
from salix import Struct

from dynamic_call_tree_resolution.composition import flatmap
from dynamic_call_tree_resolution.loader.dwarf import (
	die_name,
	every_die,
	int_value,
)
from dynamic_call_tree_resolution.loader.functions import (
	file_names,
	noreturn,
	subprogram_size,
)
from dynamic_call_tree_resolution.model import (
	Address,
	LineSpan,
	SourceLocation,
)
from dynamic_call_tree_resolution.span import Span

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator
	from pathlib import Path

	from elftools.dwarf.compileunit import CompileUnit
	from elftools.dwarf.die import DIE
	from elftools.dwarf.dwarfinfo import DWARFInfo
	from elftools.dwarf.lineprogram import LineState
	from elftools.dwarf.ranges import RangeLists


def line_spans(path: Path, locations: frozenset[SourceLocation]) -> tuple[LineSpan, ...]:
	"""The code the line table places at any of the locations."""
	with path.open("rb") as stream:
		elf = ELFFile(stream)
		dwarf = elf.get_dwarf_info(relocate_dwarf_sections=False) if elf.has_dwarf_info() else None
		spans = (
			flatmap(partial(_unit_line_spans, dwarf), dwarf.iter_CUs()) if dwarf is not None else ()
		)
		return tuple(span for span in spans if span.location in locations)


def _unit_line_spans(dwarf: DWARFInfo, unit: CompileUnit) -> Iterator[LineSpan]:
	line_program = dwarf.line_program_for_CU(unit)
	entries = line_program.get_entries() if line_program is not None else ()
	rows = tuple(entry.state for entry in entries if entry.state is not None)
	consecutive = flatmap(pairwise, _sequences(rows))
	names = file_names(dwarf, unit)
	return (
		LineSpan(
			start=Address(row.address),
			end=Address(following.address),
			location=SourceLocation(file=names[row.file], line=row.line, column=row.column),
		)
		for row, following in consecutive
		if row.file in names and following.address > row.address
	)


def _sequences(rows: tuple[LineState, ...]) -> tuple[tuple[LineState, ...], ...]:
	return tuple(
		rows[previous + 1 : end + 1]
		for previous, end in pairwise(
			(-1, *(index for index, row in enumerate(rows) if row.end_sequence))
		)
	)


class _NamedSpan(Struct, order=True):
	"""Code inlined from a function, by the function's name."""

	name: str
	span: Span


def inlined(dwarf: DWARFInfo | None) -> dict[str, tuple[Span, ...]]:
	if dwarf is None:
		return {}
	dies = tuple(
		die
		for die in every_die(dwarf)
		if die.tag in ("DW_TAG_subprogram", "DW_TAG_inlined_subroutine")
	)
	abstract = {
		die.offset: die_name(die)
		for die in dies
		if die.tag == "DW_TAG_subprogram" and "DW_AT_inline" in die.attributes
	}
	copies = (die for die in dies if die.tag == "DW_TAG_inlined_subroutine")
	named_spans = sorted(flatmap(partial(_copy_spans, dwarf, abstract), copies))
	return {
		name: tuple(named.span for named in group)
		for name, group in groupby(named_spans, key=_span_name)
	}


def _copy_spans(dwarf: DWARFInfo, abstract: dict[int, str], copy: DIE) -> Iterable[_NamedSpan]:
	origin = _reference(copy.cu, copy.attributes.get("DW_AT_abstract_origin"))
	return (
		(_NamedSpan(name=abstract[origin], span=span) for span in _die_spans(dwarf, copy.cu, copy))
		if origin in abstract
		else ()
	)


def noreturn_inlines(dwarf: DWARFInfo | None) -> tuple[tuple[Span, ...], ...]:
	"""The code spans of each inlined copy of a function DWARF declares never returns."""
	if dwarf is None:
		return ()
	copies = (
		die for die in every_die(dwarf) if die.tag == "DW_TAG_inlined_subroutine" and noreturn(die)
	)
	copy_spans = (_die_spans(dwarf, copy.cu, copy) for copy in copies)
	return tuple(spans for spans in copy_spans if spans)


def _span_name(named_span: _NamedSpan) -> str:
	return named_span.name


def _reference(unit: CompileUnit, attribute: AttributeValue | None) -> int | None:
	match attribute:
		case None:
			return None  # pragma: no cover
		case AttributeValue(form="DW_FORM_ref_addr"):
			return int_value(attribute)  # pragma: no cover
		case AttributeValue():
			return unit.cu_offset + int_value(attribute)
		case _ as unreachable:
			assert_never(unreachable)


def _die_spans(dwarf: DWARFInfo, unit: CompileUnit, die: DIE) -> tuple[Span, ...]:
	match die.attributes.get("DW_AT_ranges"), die.attributes.get("DW_AT_low_pc"):
		case AttributeValue() as ranges, _:
			return _range_spans(dwarf.range_lists(), int_value(ranges), unit)
		case None, AttributeValue() as low_pc:
			return (
				Span(
					start=Address(int_value(low_pc)),
					end=Address(int_value(low_pc) + subprogram_size(die)),
				),
			)
		case _:
			return ()  # pragma: no cover


def _range_spans(
	range_lists: RangeLists | None, offset: int, unit: CompileUnit
) -> tuple[Span, ...]:
	if range_lists is None:
		return ()  # pragma: no cover
	entries = range_lists.get_range_list_at_offset(offset, cu=unit)
	return tuple(
		Span(start=Address(entry.begin_offset), end=Address(entry.end_offset))
		if entry.is_absolute
		else Span(start=Address(base + entry.begin_offset), end=Address(base + entry.end_offset))
		for entry, base in zip(
			entries, accumulate(entries, _range_base, initial=_unit_base(unit)), strict=False
		)
		if isinstance(entry, RangeEntry)
	)


def _range_base(base: int, entry: RangeEntry | BaseAddressEntry) -> int:
	return entry.base_address if isinstance(entry, BaseAddressEntry) else base


def _unit_base(unit: CompileUnit) -> int:
	low_pc = unit.get_top_DIE().attributes.get("DW_AT_low_pc")
	return int_value(low_pc) if low_pc is not None else 0
