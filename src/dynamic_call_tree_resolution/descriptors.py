# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""What indirect calls load their callees from and what stores write (#158, #341)."""

from __future__ import annotations

from itertools import groupby
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, assert_never

from salix import Struct

from dynamic_call_tree_resolution.callgraph import IndirectCall, indirect_calls
from dynamic_call_tree_resolution.model import Field, SourceLocation
from dynamic_call_tree_resolution.stack_analysis import frame_key

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path


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


class EveryMember(Struct):
	"""Every member of a record, as a whole-record store or a write through a pointer to it writes."""

	record: str


class Unnamed(Struct):
	"""What a store that names no field or variable writes, such as ``*out = v`` through an ``int *``."""


class Unknown(Struct):
	"""What a store the unit's types can't place may write: anything."""


type WriteTarget = Field | Variable | EveryMember | Unnamed | Unknown
"""A ``variable`` is a symbol, except that a function-scope static sharing its name with another
keeps the bare name (#347): it stands for each ``name.N``."""


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


class Write(Struct):
	"""A store a statement makes, or a builtin's write through an argument."""

	unit: str
	function: str
	location: SourceLocation
	target: WriteTarget


class Pass(Struct):
	"""A pointer a call hands a callee without a dump, an unknown callee, or inline assembly."""

	unit: str
	function: str
	location: SourceLocation
	callee: str
	"""The callee's name, ``<indirect>`` or ``<asm>``."""
	argument: int
	target: WriteTarget


class Descriptors(Struct):
	"""One build's sites and stores, and what its stores write."""

	sites: tuple[Site, ...]
	stores: tuple[Store, ...]
	writes: tuple[Write, ...] = ()
	passes: tuple[Pass, ...] = ()
	taken: tuple[WriteTarget, ...] = ()
	"""The scalar fields and variables whose address the build takes: all a store that names no
	field can reach."""
	every_store: bool = False
	"""Whether the file records every store, as ``dctr descriptors`` writes it, and not only the
	plugin's stores of a function pointer."""


class _EveryStore(Struct):
	"""The record that marks a file as recording every store."""


class _Taken(Struct):
	"""A scalar field or variable whose address the build takes."""

	target: WriteTarget


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
	                          place=Variable(symbol='_char_out'), value=Parameter(name='fn')),),
	            writes=(), passes=(), taken=(), every_store=False)
	"""
	records = tuple(_record(line.split("\t")) for line in text.splitlines() if line)
	return Descriptors(
		sites=tuple(record for record in records if isinstance(record, Site)),
		stores=tuple(record for record in records if isinstance(record, Store)),
		writes=tuple(record for record in records if isinstance(record, Write)),
		passes=tuple(record for record in records if isinstance(record, Pass)),
		taken=tuple(record.target for record in records if isinstance(record, _Taken)),
		every_store=any(isinstance(record, _EveryStore) for record in records),
	)


def described_calls(descriptors: Descriptors, build_directory: Path) -> tuple[DescribedCall, ...]:
	return tuple(
		DescribedCall(
			unit=unit,
			call=call,
			callee=_agreed(
				callees.get(
					_CallKey(unit=unit, function=frame_key(call.caller), location=call.location),
					frozenset(),
				)
			),
		)
		for callees in (_callees(descriptors.sites),)
		for path in sorted(build_directory.glob("**/*.ci"))
		for unit in (path.relative_to(build_directory).with_suffix("").as_posix(),)
		for call in indirect_calls(path)
	)


class _CallKey(Struct, order=True):
	"""Where an indirect call is, by unit, frame key and source location."""

	unit: str
	function: str
	location: SourceLocation


def _callees(sites: tuple[Site, ...]) -> Mapping[_CallKey, frozenset[Descriptor]]:
	return {
		key: frozenset(site.callee for site in group)
		for key, group in groupby(sorted(sites, key=_call_key), key=_call_key)
	}


def _call_key(site: Site) -> _CallKey:
	return _CallKey(unit=site.unit, function=frame_key(site.function), location=site.location)


def _agreed(callees: frozenset[Descriptor]) -> Descriptor | None:
	match tuple(callees):
		case (callee,):
			return callee
		case _:
			return None


def _record(fields: list[str]) -> Site | Store | Write | Pass | _Taken | _EveryStore:
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
		case ["write", unit, function, file, line, column, *target]:
			return Write(
				unit=unit,
				function=function,
				location=_location(file, line, column),
				target=_write_target(target),
			)
		case ["pass", unit, function, file, line, column, callee, argument, *target]:
			return Pass(
				unit=unit,
				function=function,
				location=_location(file, line, column),
				callee=callee,
				argument=int(argument),
				target=_write_target(target),
			)
		case ["taken", *target]:
			return _Taken(target=_write_target(target))
		case ["writes", "complete"]:
			return _EveryStore()
		case _:
			raise ValueError(f"unknown descriptor record {fields}")


def _location(file: str, line: str, column: str) -> SourceLocation:
	return SourceLocation(file=PurePosixPath(file).name, line=int(line), column=int(column))


def descriptor_fields(descriptor: Descriptor) -> tuple[str, ...]:
	"""A descriptor as the plugin's records write it, which the parser reads back.

	>>> all(
	...     _descriptor(list(descriptor_fields(descriptor))) == descriptor
	...     for descriptor in (
	...         Field(record="struct ops", member="run"),
	...         Variable(symbol="hook"),
	...         Parameter(name="fn"),
	...         Element(array="handlers"),
	...         Function(symbol="char_out"),
	...         Null(),
	...         Other(type="void (*<T1>) (void)"),
	...     )
	... )
	True
	"""
	match descriptor:
		case Field(record=record, member=member):
			return ("field", record, member)
		case Variable(symbol=symbol):
			return ("variable", symbol)
		case Parameter(name=name):
			return ("parameter", name)
		case Element(array=array):
			return ("array", array)
		case Function(symbol=symbol):
			return ("function", symbol)
		case Null():
			return ("null", "0")
		case Other(type=type_):
			return ("other", type_)
		case _ as unreachable:
			assert_never(unreachable)


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


def write_target_fields(target: WriteTarget) -> tuple[str, ...]:
	"""A write target as ``dctr descriptors`` writes it, which the parser reads back.

	>>> all(
	...     _write_target(list(write_target_fields(target))) == target
	...     for target in (
	...         Field(record="struct ops", member="run"),
	...         Variable(symbol="hook.0"),
	...         EveryMember(record="struct k_thread"),
	...         Unnamed(),
	...         Unknown(),
	...     )
	... )
	True
	"""
	match target:
		case Field(record=record, member=member):
			return ("field", record, member)
		case Variable(symbol=symbol):
			return ("variable", symbol)
		case EveryMember(record=record):
			return ("record", record)
		case Unnamed():
			return ("none",)
		case Unknown():
			return ("unknown",)
		case _ as unreachable:
			assert_never(unreachable)


def _write_target(fields: list[str]) -> WriteTarget:
	"""What a store writes.

	>>> _write_target(["array", "handlers"])
	Traceback (most recent call last):
	ValueError: unknown write target ['array', 'handlers']

	Raises:
		ValueError: for a target ``dctr descriptors`` does not write.
	"""
	match fields:
		case ["field", record, member]:
			return Field(record=record, member=member)
		case ["variable", symbol]:
			return Variable(symbol=symbol)
		case ["record", record]:
			return EveryMember(record=record)
		case ["none"]:
			return Unnamed()
		case ["unknown"]:
			return Unknown()
		case _:
			raise ValueError(f"unknown write target {fields}")
