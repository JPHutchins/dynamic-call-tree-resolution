# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Parsing ``-fstack-usage`` (``.su``) records, from GCC or clang."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Final, assert_never

from salix import Struct

from dynamic_call_tree_resolution.model import SourceLocation

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path


class StackUsage(Struct):
	"""One ``-fstack-usage`` record."""

	function: str
	bytes: int
	bounded: bool
	"""``False`` for a plain ``dynamic`` frame (``alloca`` or a variable-length array)."""


def parse_stack_usage(path: Path) -> tuple[StackUsage, ...]:
	return parse_records(path.read_text())


def parse_records(text: str) -> tuple[StackUsage, ...]:
	"""Parse ``.su`` lines into records.

	>>> parse_records(
	...     "main.c:1:1:main" + chr(9) + "16" + chr(9) + "static" + chr(10)
	...     + "log.c:2:1:log" + chr(9) + "24" + chr(9) + "dynamic,bounded" + chr(10)
	...     + "worker.c:3:1:worker" + chr(9) + "32" + chr(9) + "dynamic" + chr(10)
	... )  # doctest: +NORMALIZE_WHITESPACE
	(StackUsage(function='main', bytes=16, bounded=True),
	 StackUsage(function='log', bytes=24, bounded=True),
	 StackUsage(function='worker', bytes=32, bounded=False))
	"""
	return tuple(_parse_record(line) for line in text.splitlines() if line)


def load_stack_usages(build_directory: Path) -> tuple[StackUsage, ...]:
	return tuple(record for _, records in stack_usage_files(build_directory) for record in records)


def stack_usage_files(
	build_directory: Path,
) -> tuple[tuple[Path, tuple[StackUsage, ...]], ...]:
	return tuple(
		(stack_file, parse_stack_usage(stack_file))
		for stack_file in sorted(build_directory.glob("**/*.su"))
	)


def stack_usage_locations(path: Path) -> Mapping[str, SourceLocation]:
	return dict(map(_record_location, filter(None, path.read_text().splitlines())))


def _record_location(line: str) -> tuple[str, SourceLocation]:
	"""Where a ``.su`` record says its function is declared, by the function's name.

	>>> _record_location("/src/kernel/sched.c:403:13:reschedule" + chr(9) + "0" + chr(9) + "static")
	('reschedule', SourceLocation(file='sched.c', line=403, column=13))
	"""
	location = _location(line.split("\t", maxsplit=1)[0])
	return location.function, SourceLocation(
		file=PurePosixPath(location.path).name, line=location.line, column=location.column
	)


_LOCATION: Final = re.compile(r"(?P<path>.*?):(?P<line>\d+):(?:(?P<column>\d+):)?(?P<function>.+)")


class _Location(Struct):
	path: str
	line: int
	column: int
	"""0 when the record has none, as clang's don't."""
	function: str


def _location(text: str) -> _Location:
	"""The parts of a ``.su`` record's location.

	>>> _location("vla.c:1:big")
	_Location(path='vla.c', line=1, column=0, function='big')
	>>> _location("a.cc:3:6:void Foo::bar()")
	_Location(path='a.cc', line=3, column=6, function='void Foo::bar()')

	Raises:
		ValueError: for a location without a line number.
	"""
	match _LOCATION.fullmatch(text):
		case None:
			raise ValueError(f"malformed .su location {text!r}")
		case re.Match() as found:
			return _Location(
				path=found["path"],
				line=int(found["line"]),
				column=int(found["column"] or 0),
				function=found["function"],
			)
		case _ as unreachable:
			assert_never(unreachable)


def _parse_record(line: str) -> StackUsage:
	location, byte_count, qualifier = line.split("\t")
	return StackUsage(
		function=_location(location).function,
		bytes=int(byte_count),
		bounded=_bounded(qualifier),
	)


def _bounded(qualifier: str) -> bool:
	"""Whether the bytes of a ``.su`` record bound its frame.

	Raises:
		ValueError: for anything GCC does not emit.
	"""
	match qualifier:
		case "static" | "dynamic,bounded":
			return True
		case "dynamic":
			return False
		case _:
			raise ValueError(f"unknown .su qualifier {qualifier!r}")
