# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Source lines and inlined copies, as address spans."""

from __future__ import annotations

from itertools import accumulate, groupby, pairwise
from typing import TYPE_CHECKING, assert_never

from elftools.dwarf.die import DIE, AttributeValue
from elftools.dwarf.ranges import BaseAddressEntry, RangeEntry
from elftools.elf.elffile import ELFFile
from salix import Struct

from dynamic_call_tree_resolution.loader.dwarf import (
	die_name,
	int_value,
	iter_dies,
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
	from pathlib import Path

	from elftools.dwarf.compileunit import CompileUnit
	from elftools.dwarf.dwarfinfo import DWARFInfo
	from elftools.dwarf.lineprogram import LineState
	from elftools.dwarf.ranges import RangeLists


def line_spans(path: Path, locations: frozenset[SourceLocation]) -> tuple[LineSpan, ...]:
	"""The code the line table places at any of the locations."""
	with path.open("rb") as stream:
		elf = ELFFile(stream)
		dwarf = elf.get_dwarf_info(relocate_dwarf_sections=False) if elf.has_dwarf_info() else None
		return (
			tuple(
				span
				for unit in dwarf.iter_CUs()
				for line_program in (dwarf.line_program_for_CU(unit),)
				if line_program is not None
				for names in (file_names(dwarf, unit),)
				for sequence in _sequences(
					tuple(
						entry.state
						for entry in line_program.get_entries()
						if entry.state is not None
					)
				)
				for row, following in pairwise(sequence)
				if row.file in names and following.address > row.address
				for span in (
					LineSpan(
						start=Address(row.address),
						end=Address(following.address),
						location=SourceLocation(
							file=names[row.file], line=row.line, column=row.column
						),
					),
				)
				if span.location in locations
			)
			if dwarf is not None
			else ()
		)


def _sequences(rows: tuple[LineState, ...]) -> tuple[tuple[LineState, ...], ...]:
	return tuple(
		rows[previous + 1 : end + 1]
		for previous, end in pairwise(
			(-1, *(index for index, row in enumerate(rows) if row.end_sequence))
		)
	)


class _UnitDie(Struct):
	"""A DIE, with the compile unit that holds it."""

	unit: CompileUnit
	die: DIE


class _NamedSpan(Struct, order=True):
	"""Code inlined from a function, by the function's name."""

	name: str
	span: Span


def inlined(dwarf: DWARFInfo | None) -> dict[str, tuple[Span, ...]]:
	if dwarf is None:
		return {}
	dies = tuple(
		_UnitDie(unit=unit, die=die)
		for unit in dwarf.iter_CUs()
		for die in iter_dies(unit.get_top_DIE())
		if die.tag in ("DW_TAG_subprogram", "DW_TAG_inlined_subroutine")
	)
	abstract = {
		found.die.offset: die_name(found.die)
		for found in dies
		if found.die.tag == "DW_TAG_subprogram" and "DW_AT_inline" in found.die.attributes
	}
	return {
		name: tuple(named.span for named in group)
		for name, group in groupby(
			sorted(
				_NamedSpan(name=abstract[origin], span=span)
				for found in dies
				if found.die.tag == "DW_TAG_inlined_subroutine"
				for origin in (
					_reference(found.unit, found.die.attributes.get("DW_AT_abstract_origin")),
				)
				if origin in abstract
				for span in _die_spans(dwarf, found.unit, found.die)
			),
			key=_span_name,
		)
	}


def noreturn_inlines(dwarf: DWARFInfo | None) -> tuple[tuple[Span, ...], ...]:
	"""The code spans of each inlined copy of a function DWARF declares never returns."""
	if dwarf is None:
		return ()
	return tuple(
		spans
		for unit in dwarf.iter_CUs()
		for die in iter_dies(unit.get_top_DIE())
		if die.tag == "DW_TAG_inlined_subroutine" and noreturn(die)
		if (spans := _die_spans(dwarf, unit, die))
	)


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
