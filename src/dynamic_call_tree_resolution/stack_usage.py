# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Parsing GCC ``-fstack-usage`` (``.su``) records."""

from __future__ import annotations

from typing import TYPE_CHECKING

from salix import Struct

if TYPE_CHECKING:
	from pathlib import Path


class StackUsage(Struct):
	"""One ``-fstack-usage`` record: a function's stack frame."""

	function: str
	bytes: int
	dynamic: bool


def parse_stack_usage(path: Path) -> tuple[StackUsage, ...]:
	"""Parse one GCC ``.su`` file into its records.

	Record layout: ``file:line:column:function<TAB>bytes<TAB>qualifier``
	where ``qualifier`` is ``static`` for fixed frames and ``dynamic`` for
	frames using ``alloca`` or variable-length arrays.
	"""
	return tuple(_parse_record(line) for line in path.read_text().splitlines() if line)


def load_stack_usages(build_directory: Path) -> tuple[StackUsage, ...]:
	"""Collect every ``.su`` record under a build directory."""
	return tuple(
		record
		for stack_file in sorted(build_directory.glob("**/*.su"))
		for record in parse_stack_usage(stack_file)
	)


def _parse_record(line: str) -> StackUsage:
	location, byte_count, qualifier = line.split("\t")
	return StackUsage(
		function=location.split(":", 3)[3],
		bytes=int(byte_count),
		dynamic=qualifier == "dynamic",
	)
