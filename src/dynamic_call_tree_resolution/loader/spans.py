# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Source lines and inlined copies, as address spans."""

from __future__ import annotations

from itertools import accumulate, groupby, pairwise
from typing import TYPE_CHECKING, assert_never

from elftools.dwarf.die import DIE, AttributeValue
from elftools.dwarf.ranges import BaseAddressEntry, RangeEntry
from elftools.elf.elffile import ELFFile

from dynamic_call_tree_resolution.loader.dwarf import (
	die_name,
	int_value,
	iter_dies,
)
from dynamic_call_tree_resolution.loader.functions import (
	file_names,
	subprogram_size,
)
from dynamic_call_tree_resolution.model import (
	Address,
	LineSpan,
	SourceLocation,
)

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


def inlined(dwarf: DWARFInfo | None) -> dict[str, tuple[tuple[Address, Address], ...]]:
	if dwarf is None:
		return {}
	dies = tuple(
		(unit, die)
		for unit in dwarf.iter_CUs()
		for die in iter_dies(unit.get_top_DIE())
		if die.tag in ("DW_TAG_subprogram", "DW_TAG_inlined_subroutine")
	)
	abstract = {
		die.offset: die_name(die)
		for _, die in dies
		if die.tag == "DW_TAG_subprogram" and "DW_AT_inline" in die.attributes
	}
	return {
		name: tuple(span for _, span in group)
		for name, group in groupby(
			sorted(
				(abstract[origin], span)
				for unit, die in dies
				if die.tag == "DW_TAG_inlined_subroutine"
				for origin in (_reference(unit, die.attributes.get("DW_AT_abstract_origin")),)
				if origin in abstract
				for span in _die_spans(dwarf, unit, die)
			),
			key=_span_name,
		)
	}


def _span_name(named_span: tuple[str, tuple[Address, Address]]) -> str:
	return named_span[0]


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


def _die_spans(
	dwarf: DWARFInfo, unit: CompileUnit, die: DIE
) -> tuple[tuple[Address, Address], ...]:
	match die.attributes.get("DW_AT_ranges"), die.attributes.get("DW_AT_low_pc"):
		case AttributeValue() as ranges, _:
			return _range_spans(dwarf.range_lists(), int_value(ranges), unit)
		case None, AttributeValue() as low_pc:
			return (
				(
					Address(int_value(low_pc)),
					Address(int_value(low_pc) + subprogram_size(die)),
				),
			)
		case _:
			return ()  # pragma: no cover


def _range_spans(
	range_lists: RangeLists | None, offset: int, unit: CompileUnit
) -> tuple[tuple[Address, Address], ...]:
	if range_lists is None:
		return ()  # pragma: no cover
	entries = range_lists.get_range_list_at_offset(offset, cu=unit)
	return tuple(
		(Address(begin), Address(end))
		for entry, base in zip(
			entries, accumulate(entries, _range_base, initial=_unit_base(unit)), strict=False
		)
		if isinstance(entry, RangeEntry)
		for begin, end in (
			(entry.begin_offset, entry.end_offset)
			if entry.is_absolute
			else (base + entry.begin_offset, base + entry.end_offset),
		)
	)


def _range_base(base: int, entry: RangeEntry | BaseAddressEntry) -> int:
	return entry.base_address if isinstance(entry, BaseAddressEntry) else base


def _unit_base(unit: CompileUnit) -> int:
	low_pc = unit.get_top_DIE().attributes.get("DW_AT_low_pc")
	return int_value(low_pc) if low_pc is not None else 0
