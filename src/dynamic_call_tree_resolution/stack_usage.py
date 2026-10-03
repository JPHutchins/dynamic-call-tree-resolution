# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Parsing GCC ``-fstack-usage`` (``.su``) records."""

from __future__ import annotations

from typing import TYPE_CHECKING

from salix import Struct

if TYPE_CHECKING:
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


def _parse_record(line: str) -> StackUsage:
	location, byte_count, qualifier = line.split("\t")
	return StackUsage(
		function=location.split(":", 3)[3],
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
