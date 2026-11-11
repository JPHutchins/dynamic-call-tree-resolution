# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The descriptors plugin's records, from the alias dumps a build's GCC wrote (#205)."""

from __future__ import annotations

from functools import partial
from itertools import chain
from typing import TYPE_CHECKING

from salix import Struct

from dynamic_call_tree_resolution.composition import Pipeline, flatmap
from dynamic_call_tree_resolution.gcc_dumps.dump import DumpFunction, dump_functions
from dynamic_call_tree_resolution.gcc_dumps.records import (
	Gap,
	SiteRecord,
	StoreRecord,
	record_line,
	unit_records,
)
from dynamic_call_tree_resolution.gcc_dumps.taken import taken_line, unit_taken
from dynamic_call_tree_resolution.gcc_dumps.units import DUMP_FLAG as DUMP_FLAG
from dynamic_call_tree_resolution.gcc_dumps.units import (
	DumpedUnit,
	defined_functions,
	dumped_units,
	unit_types,
)
from dynamic_call_tree_resolution.gcc_dumps.writes import EVERY_STORE, unit_writes, write_line

if TYPE_CHECKING:
	from collections.abc import Iterator
	from pathlib import Path

	from dynamic_call_tree_resolution.descriptors import WriteTarget
	from dynamic_call_tree_resolution.gcc_dumps.records import DumpRecord
	from dynamic_call_tree_resolution.gcc_dumps.units import UnitTypes
	from dynamic_call_tree_resolution.gcc_dumps.writes import WriteOrPass


class DumpDescriptors(Struct):
	"""A build's descriptor records, and the statements no record could be made of."""

	lines: tuple[str, ...]
	"""Sorted, as the plugin's replay writes ``descriptors.txt``."""
	writes: tuple[str, ...]
	"""What every store writes, what calls hand code without a dump, and the scalar fields and
	variables whose address is taken, each sorted, then the line that marks the file as recording
	every store: a file cut short lacks it."""
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


class _TypedUnit(Struct):
	"""A dumped unit, with the types its object describes."""

	parsed: _ParsedUnit
	types: UnitTypes


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
	dumped = Pipeline(units).then(partial(flatmap, _dumped_names)).then(frozenset).value
	typed = tuple(
		_TypedUnit(parsed=unit, types=unit_types(unit.dumped.object))
		for unit in units
		if unit.dumped.dump is not None
	)
	records = tuple(flatmap(partial(_typed_records, root), typed))
	return DumpDescriptors(
		units=len(units),
		dumps=len(typed),
		lines=tuple(
			sorted(
				record_line(record)
				for record in records
				if isinstance(record, SiteRecord | StoreRecord)
			)
		),
		writes=(
			*sorted(
				write_line(record)
				for record in flatmap(partial(_typed_writes, root, dumped), typed)
			),
			*sorted({taken_line(target) for target in flatmap(_typed_taken, typed)}),
			EVERY_STORE,
		),
		gaps=tuple(record for record in records if isinstance(record, Gap)),
	)


def _dumped_names(unit: _ParsedUnit) -> Iterator[str]:
	symbols = frozenset(function.symbol for function in unit.functions)
	return chain.from_iterable(
		aliases for aliases in chain(unit.defined, (symbols,)) if not aliases.isdisjoint(symbols)
	)


def _typed_records(root: Path, unit: _TypedUnit) -> tuple[DumpRecord, ...]:
	return unit_records(unit.parsed.dumped.unit, unit.parsed.functions, unit.types, root)


def _typed_writes(root: Path, dumped: frozenset[str], unit: _TypedUnit) -> Iterator[WriteOrPass]:
	return unit_writes(unit.parsed.dumped.unit, unit.parsed.functions, unit.types, root, dumped)


def _typed_taken(unit: _TypedUnit) -> Iterator[WriteTarget]:
	return unit_taken(unit.parsed.functions, unit.types)


def _check_functions(units: tuple[_ParsedUnit, ...]) -> None:
	"""Check that each unit's dump has exactly the functions its object defines.

	Raises:
		ValueError: when a defined function has no dump, as GCC dumps only what it optimizes, or a
			dumped function is not defined, as in a stale dump.
	"""
	undumped = tuple(flatmap(_undumped, units))
	if undumped:
		raise ValueError(
			f"{len(undumped)} functions have no alias dump, such as {undumped[0]}: GCC dumps "
			"only the functions it optimizes"
		)
	stale = tuple(flatmap(_stale, units))
	if stale:
		raise ValueError(
			f"{len(stale)} dumped functions are not in their objects, such as {stale[0]}: the "
			"dumps are stale, rebuild"
		)


def _undumped(unit: _ParsedUnit) -> Iterator[str]:
	symbols = frozenset(function.symbol for function in unit.functions)
	return (
		f"{min(aliases)} of {unit.dumped.unit}"
		for aliases in unit.defined
		if aliases.isdisjoint(symbols)
	)


def _stale(unit: _ParsedUnit) -> Iterator[str]:
	return (
		f"{function.symbol} of {unit.dumped.unit}"
		for function in unit.functions
		if not any(function.symbol in aliases for aliases in unit.defined)
	)
