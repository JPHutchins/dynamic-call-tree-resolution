# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The descriptors plugin's records, from the alias dumps a build's GCC wrote (#205)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from salix import Struct

from dynamic_call_tree_resolution.gcc_dumps.dump import DumpFunction, dump_functions
from dynamic_call_tree_resolution.gcc_dumps.records import (
	Gap,
	SiteRecord,
	StoreRecord,
	record_line,
	unit_records,
)
from dynamic_call_tree_resolution.gcc_dumps.units import DUMP_FLAG as DUMP_FLAG
from dynamic_call_tree_resolution.gcc_dumps.units import (
	DumpedUnit,
	defined_functions,
	dumped_units,
	unit_types,
)

if TYPE_CHECKING:
	from pathlib import Path


class DumpDescriptors(Struct):
	"""A build's descriptor records, and the statements no record could be made of."""

	lines: tuple[str, ...]
	"""Sorted, as the plugin's replay writes ``descriptors.txt``."""
	gaps: tuple[Gap, ...]
	units: int
	"""The build's C units."""
	dumps: int
	"""Those GCC wrote an alias dump for: a unit without a function has none."""


class _ParsedUnit(Struct):
	"""A unit, with the functions of its alias dump and those its object defines."""

	dumped: DumpedUnit
	functions: tuple[DumpFunction, ...]
	defined: tuple[frozenset[str], ...]
	"""Each function as the names of its aliases; none for a unit the build never compiled."""


def dump_descriptors(build_directory: Path, root: Path) -> DumpDescriptors:
	"""A build's descriptor records, from every unit's alias dump and object."""
	units = tuple(
		_ParsedUnit(
			dumped=unit,
			functions=dump_functions(unit.dump.read_text()) if unit.dump is not None else (),
			defined=(
				defined_functions(unit.object)
				if unit.dump is not None or unit.object.exists()
				else ()
			),
		)
		for unit in dumped_units(build_directory)
	)
	_check_functions(units)
	records = tuple(
		record
		for unit in units
		if unit.dumped.dump is not None
		for record in unit_records(
			unit.dumped.unit, unit.functions, unit_types(unit.dumped.object), root
		)
	)
	return DumpDescriptors(
		units=len(units),
		dumps=sum(unit.dumped.dump is not None for unit in units),
		lines=tuple(
			sorted(
				record_line(record)
				for record in records
				if isinstance(record, SiteRecord | StoreRecord)
			)
		),
		gaps=tuple(record for record in records if isinstance(record, Gap)),
	)


def _check_functions(units: tuple[_ParsedUnit, ...]) -> None:
	"""Check that each unit's dump has exactly the functions its object defines.

	Raises:
		ValueError: when a defined function has no dump, as GCC dumps only what it optimizes, or a
			dumped function is not defined, as in a stale dump.
	"""
	undumped = tuple(
		f"{min(aliases)} of {unit.dumped.unit}"
		for unit in units
		for aliases in unit.defined
		if aliases.isdisjoint(function.symbol for function in unit.functions)
	)
	if undumped:
		raise ValueError(
			f"{len(undumped)} functions have no alias dump, such as {undumped[0]}: GCC dumps "
			"only the functions it optimizes"
		)
	stale = tuple(
		f"{function.symbol} of {unit.dumped.unit}"
		for unit in units
		for function in unit.functions
		if not any(function.symbol in aliases for aliases in unit.defined)
	)
	if stale:
		raise ValueError(
			f"{len(stale)} dumped functions are not in their objects, such as {stale[0]}: the "
			"dumps are stale, rebuild"
		)
