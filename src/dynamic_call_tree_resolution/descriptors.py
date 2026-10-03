# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""What each indirect call loads its callee from, and every store of a function pointer (#158)."""

from __future__ import annotations

from itertools import groupby
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from salix import Struct

from dynamic_call_tree_resolution.callgraph import IndirectCall, indirect_calls
from dynamic_call_tree_resolution.model import SourceLocation
from dynamic_call_tree_resolution.stack_analysis import frame_key

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path


class Field(Struct):
	"""A member of a struct or union."""

	record: str
	"""``struct`` or ``union``, then the type's name."""
	member: str


class Variable(Struct):
	"""A variable, by its symbol."""

	symbol: str


class Parameter(Struct):
	"""A parameter of the function holding the site or the store."""

	name: str


class Element(Struct):
	"""An element of an array."""

	array: str


class Function(Struct):
	"""A function's address, by its symbol."""

	symbol: str


class Null(Struct):
	"""A null pointer."""


class Other(Struct):
	"""Anything else, by its type as GCC prints it."""

	type: str


type Descriptor = Field | Variable | Parameter | Element | Function | Null | Other
type Place = Field | Variable


class Site(Struct):
	"""An indirect call as the descriptors plugin saw it."""

	unit: str
	"""The build's path of the unit's object, without its suffix; its ``.ci`` shares it."""
	function: str
	location: SourceLocation
	callee: Descriptor


class Store(Struct):
	"""A store of a function pointer into a field or a variable."""

	unit: str
	"""The build's path of the unit's object, without its suffix; its ``.ci`` shares it."""
	function: str
	location: SourceLocation
	place: Place
	value: Descriptor


class Descriptors(Struct):
	"""One build's sites and stores."""

	sites: tuple[Site, ...]
	stores: tuple[Store, ...]


class DescribedCall(Struct):
	"""A ``.ci`` indirect call and what the plugin says it loads its callee from."""

	unit: str
	"""The build's path of the ``.ci``, without its suffix."""
	call: IndirectCall
	callee: Descriptor | None
	"""``None`` unless every plugin site at the call agrees."""


def load_descriptors(path: Path) -> Descriptors:
	return parse_descriptors(path.read_text())


def parse_descriptors(text: str) -> Descriptors:
	"""Parse the plugin's tab-separated records.

	>>> parse_descriptors(
	...     chr(9).join(("site", "kernel/device.c", "do_device_init", "kernel/device.c", "23", "8"))
	...     + chr(9).join(("", "field", "struct device_ops", "init"))
	...     + chr(10)
	...     + chr(9).join(("store", "lib/os/printk.c", "__printk_hook_install", "lib/os/printk.c"))
	...     + chr(9).join(("", "59", "12", "variable", "_char_out", "parameter", "fn"))
	... )  # doctest: +NORMALIZE_WHITESPACE
	Descriptors(sites=(Site(unit='kernel/device.c', function='do_device_init',
	                        location=SourceLocation(file='device.c', line=23, column=8),
	                        callee=Field(record='struct device_ops', member='init')),),
	            stores=(Store(unit='lib/os/printk.c', function='__printk_hook_install',
	                          location=SourceLocation(file='printk.c', line=59, column=12),
	                          place=Variable(symbol='_char_out'), value=Parameter(name='fn')),))
	"""
	records = tuple(_record(line.split("\t")) for line in text.splitlines() if line)
	return Descriptors(
		sites=tuple(record for record in records if isinstance(record, Site)),
		stores=tuple(record for record in records if isinstance(record, Store)),
	)


def described_calls(descriptors: Descriptors, build_directory: Path) -> tuple[DescribedCall, ...]:
	return tuple(
		DescribedCall(
			unit=unit,
			call=call,
			callee=_agreed(callees.get((unit, frame_key(call.caller), call.location), frozenset())),
		)
		for callees in (_callees(descriptors.sites),)
		for path in sorted(build_directory.glob("**/*.ci"))
		for unit in (path.relative_to(build_directory).with_suffix("").as_posix(),)
		for call in indirect_calls(path)
	)


def _callees(
	sites: tuple[Site, ...],
) -> Mapping[tuple[str, str, SourceLocation], frozenset[Descriptor]]:
	return {
		key: frozenset(site.callee for site in group)
		for key, group in groupby(sorted(sites, key=_site_order), key=_site_key)
	}


def _site_key(site: Site) -> tuple[str, str, SourceLocation]:
	return (site.unit, frame_key(site.function), site.location)


def _site_order(site: Site) -> tuple[str, str, str, int, int]:
	return (
		site.unit,
		frame_key(site.function),
		site.location.file,
		site.location.line,
		site.location.column,
	)


def _agreed(callees: frozenset[Descriptor]) -> Descriptor | None:
	match tuple(callees):
		case (callee,):
			return callee
		case _:
			return None


def _record(fields: list[str]) -> Site | Store:
	"""One plugin record.

	>>> _record(["call", "a.c", "f", "a.c", "1", "2", "null", "0"])
	Traceback (most recent call last):
	ValueError: unknown descriptor record ['call', 'a.c', 'f', 'a.c', '1', '2', 'null', '0']

	Raises:
		ValueError: for a record the plugin does not write.
	"""
	match fields:
		case ["site", unit, function, file, line, column, *callee]:
			return Site(
				unit=unit,
				function=function,
				location=_location(file, line, column),
				callee=_descriptor(callee),
			)
		case ["store", unit, function, file, line, column, "field", record, member, *value]:
			return Store(
				unit=unit,
				function=function,
				location=_location(file, line, column),
				place=Field(record=record, member=member),
				value=_descriptor(value),
			)
		case ["store", unit, function, file, line, column, "variable", symbol, *value]:
			return Store(
				unit=unit,
				function=function,
				location=_location(file, line, column),
				place=Variable(symbol=symbol),
				value=_descriptor(value),
			)
		case _:
			raise ValueError(f"unknown descriptor record {fields}")


def _location(file: str, line: str, column: str) -> SourceLocation:
	return SourceLocation(file=PurePosixPath(file).name, line=int(line), column=int(column))


def _descriptor(fields: list[str]) -> Descriptor:
	"""What a callee or a stored value was loaded from.

	>>> _descriptor(["other", "void (*<Txxxx>) (void)"]), _descriptor(["null", "0"])
	(Other(type='void (*<Txxxx>) (void)'), Null())
	>>> _descriptor(["array", "handlers"]), _descriptor(["function", "char_out"])
	(Element(array='handlers'), Function(symbol='char_out'))
	>>> _descriptor(["vtable", "x"])
	Traceback (most recent call last):
	ValueError: unknown descriptor ['vtable', 'x']

	Raises:
		ValueError: for a descriptor the plugin does not write.
	"""
	match fields:
		case ["field", record, member]:
			return Field(record=record, member=member)
		case ["variable", symbol]:
			return Variable(symbol=symbol)
		case ["parameter", name]:
			return Parameter(name=name)
		case ["array", array]:
			return Element(array=array)
		case ["function", symbol]:
			return Function(symbol=symbol)
		case ["null", "0"]:
			return Null()
		case ["other", type_]:
			return Other(type=type_)
		case _:
			raise ValueError(f"unknown descriptor {fields}")
