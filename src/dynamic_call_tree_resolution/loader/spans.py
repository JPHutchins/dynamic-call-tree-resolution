# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Source lines and inlined copies, as address spans."""

from __future__ import annotations

from bisect import bisect_left
from functools import partial
from itertools import accumulate, chain, groupby, pairwise, starmap
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
	from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
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


def line_locations(
	path: Path, addresses: Collection[Address]
) -> Mapping[Address, frozenset[SourceLocation]]:
	"""Where the line table places each address.

	An address is absent when no row covers it, when a row covering it has no source line, or when
	rows of more than one sequence cover it.
	"""
	with path.open("rb") as stream:
		elf = ELFFile(stream)
		dwarf = elf.get_dwarf_info(relocate_dwarf_sections=False) if elf.has_dwarf_info() else None
		return _line_locations(dwarf, sorted(addresses)) if dwarf is not None else {}


def _line_locations(
	dwarf: DWARFInfo, addresses: Sequence[Address]
) -> Mapping[Address, frozenset[SourceLocation]]:
	covering = sorted(
		flatmap(partial(_unit_covering, dwarf, addresses), dwarf.iter_CUs()),
		key=_covering_address,
	)
	grouped = {address: tuple(group) for address, group in groupby(covering, key=_covering_address)}
	return {
		address: frozenset(_source_lines(group))
		for address, group in grouped.items()
		if _one_sequence_with_lines(group)
	}


class _NoLine(Struct):
	"""A row that places its code at line 0, or in a file its unit doesn't list."""


class _Sequence(Struct):
	"""One sequence of a unit's line program."""

	unit: int
	"""The unit's offset in ``.debug_info``."""
	index: int


class _Covering(Struct):
	"""A row that covers an address."""

	address: Address
	sequence: _Sequence
	line: SourceLocation | _NoLine


class _RowsAt(Struct):
	"""The line table rows of one sequence that share an address."""

	address: Address
	rows: tuple[LineState, ...]


def _covering_address(covering: _Covering) -> Address:
	return covering.address


def _one_sequence_with_lines(group: tuple[_Covering, ...]) -> bool:
	return len({covering.sequence for covering in group}) == 1 and not any(
		isinstance(covering.line, _NoLine) for covering in group
	)


def _source_lines(group: tuple[_Covering, ...]) -> Iterator[SourceLocation]:
	return (covering.line for covering in group if isinstance(covering.line, SourceLocation))


def _unit_covering(
	dwarf: DWARFInfo, addresses: Sequence[Address], unit: CompileUnit
) -> Iterator[_Covering]:
	return chain.from_iterable(
		starmap(
			partial(_sequence_covering, file_names(dwarf, unit), addresses, unit.cu_offset),
			enumerate(_sequences(_unit_rows(dwarf, unit))),
		)
	)


def _sequence_covering(
	names: dict[int, str],
	addresses: Sequence[Address],
	unit: int,
	index: int,
	rows: tuple[LineState, ...],
) -> Iterator[_Covering]:
	return chain.from_iterable(
		starmap(
			partial(_rows_covering, names, addresses, _Sequence(unit=unit, index=index)),
			pairwise(_rows_at(rows)),
		)
	)


def _rows_covering(
	names: dict[int, str],
	addresses: Sequence[Address],
	sequence: _Sequence,
	here: _RowsAt,
	following: _RowsAt,
) -> Iterator[_Covering]:
	return flatmap(
		partial(
			_row_covering,
			names,
			sequence,
			addresses[
				bisect_left(addresses, here.address) : bisect_left(addresses, following.address)
			],
		),
		here.rows,
	)


def _row_covering(
	names: dict[int, str], sequence: _Sequence, covered: Sequence[Address], row: LineState
) -> Iterator[_Covering]:
	return (
		_Covering(address=address, sequence=sequence, line=_source_line(names, row))
		for address in covered
	)


def _source_line(names: dict[int, str], row: LineState) -> SourceLocation | _NoLine:
	return (
		SourceLocation(file=names[row.file], line=row.line, column=row.column)
		if row.file in names and row.line != 0
		else _NoLine()
	)


def _rows_at(rows: tuple[LineState, ...]) -> tuple[_RowsAt, ...]:
	return tuple(
		_RowsAt(address=Address(address), rows=tuple(group))
		for address, group in groupby(rows, key=_row_address)
	)


def _row_address(row: LineState) -> int:
	return row.address


def _unit_rows(dwarf: DWARFInfo, unit: CompileUnit) -> tuple[LineState, ...]:
	line_program = dwarf.line_program_for_CU(unit)
	entries = line_program.get_entries() if line_program is not None else ()
	return tuple(entry.state for entry in entries if entry.state is not None)


def _unit_line_spans(dwarf: DWARFInfo, unit: CompileUnit) -> Iterator[LineSpan]:
	consecutive = flatmap(pairwise, _sequences(_unit_rows(dwarf, unit)))
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


def _sequences(rows: tuple[LineState, ...]) -> Iterator[tuple[LineState, ...]]:
	return (
		rows[previous + 1 : end + 1]
		for previous, end in pairwise(
			chain((-1,), (index for index, row in enumerate(rows) if row.end_sequence))
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
	copy_spans = (tuple(_die_spans(dwarf, copy.cu, copy)) for copy in copies)
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


def _die_spans(dwarf: DWARFInfo, unit: CompileUnit, die: DIE) -> Iterable[Span]:
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


def _range_spans(range_lists: RangeLists | None, offset: int, unit: CompileUnit) -> Iterable[Span]:
	if range_lists is None:
		return ()  # pragma: no cover
	entries = range_lists.get_range_list_at_offset(offset, cu=unit)
	return (
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
