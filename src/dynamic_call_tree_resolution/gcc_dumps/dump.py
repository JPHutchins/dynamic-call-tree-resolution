# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The functions of a ``-fdump-tree-alias-lineno`` dump: their parameters, declarations and statements."""

from __future__ import annotations

import re
from itertools import accumulate, pairwise
from typing import TYPE_CHECKING, Final

from salix import Struct

if TYPE_CHECKING:
	from collections.abc import Mapping

_HEADER: Final = re.compile(r"^;; Function (\S+) \((\S+?),", re.MULTILINE)
_LOCATION: Final = re.compile(r"\[([^\[\]]+?):(\d+):(\d+)(?: discrim \d+)?\] ")
_CALL_FLAGS: Final = re.compile(r";(?: \[[^\[\]]*\])*$")
_LABEL: Final = re.compile(r"^<(?:bb \d+|L\d+)>")


class Statement(Struct):
	"""A statement of a function's body, without the locations GCC prints inside it."""

	file: str
	line: int
	column: int
	body: str


class _Declared(Struct):
	"""A name a dump declares, with its type as GCC prints it."""

	name: str
	type: str


class DumpFunction(Struct):
	"""One function of a dump."""

	symbol: str
	"""Its assembler name."""
	parameters: Mapping[str, str]
	"""Each parameter's type, by its name."""
	declarations: Mapping[str, str]
	"""Each local's or SSA name's type, by the name it is declared under."""
	statements: tuple[Statement, ...]


def dump_functions(text: str) -> tuple[DumpFunction, ...]:
	r"""The functions of a dump.

	>>> dump_functions(
	...     ";; Function f (f, funcdef_no=1)\n\nint f (int (*) (int) g, int x)\n{\n"
	...     "  int _2;\n\n  <bb 2> :\n  [a.c:3:9] # DEBUG BEGIN_STMT\n"
	...     "  [a.c:3:9] _2 = [a.c:3:9] g_1(D) (x_3(D));\n  [a.c:4:1] return _2;\n}\n"
	... )  # doctest: +NORMALIZE_WHITESPACE
	(DumpFunction(symbol='f', parameters={'g': 'int (*) (int)', 'x': 'int'},
	              declarations={'_2': 'int'},
	              statements=(Statement(file='a.c', line=3, column=9, body='_2 = g_1(D) (x_3(D))'),
	                          Statement(file='a.c', line=4, column=1, body='return _2'))),)
	"""
	headers = tuple(_HEADER.finditer(text))
	return tuple(
		_function(header.group(1), header.group(2), text[header.end() : end])
		for header, end in zip(
			headers, (*(header.start() for header in headers[1:]), len(text)), strict=True
		)
	)


def _function(name: str, symbol: str, text: str) -> DumpFunction:
	lines = text.splitlines()
	opening = next(index for index, line in enumerate(lines) if line.startswith("{"))
	closing = next(
		index for index, line in enumerate(lines) if index > opening and line.startswith("}")
	)
	body = lines[opening + 1 : closing]
	declared = tuple(
		line for line in _leading(body) if line.strip() and not line.strip().startswith("<bb")
	)
	return DumpFunction(
		symbol=symbol,
		parameters=_parameters(name, next(line for line in reversed(lines[:opening]) if line)),
		declarations={declared.name: declared.type for declared in map(_declaration, declared)},
		statements=tuple(
			statement
			for line in body[len(_leading(body)) :]
			if (statement := _statement(line.strip())) is not None
		),
	)


def _leading(body: list[str]) -> list[str]:
	"""The declaration block: the lines before the first blank, block label or statement."""
	ends = (
		index
		for index, line in enumerate(body)
		if not line.strip() or line.strip().startswith("<bb") or _LOCATION.match(line.strip())
	)
	return body[: next(ends, len(body))]


def _parameters(name: str, signature: str) -> dict[str, str]:
	start = signature.index(f"{name} (") + len(name) + 2
	return {
		declared.name: declared.type
		for part in _top_level_parts(signature[start : signature.rindex(")")])
		if part and part != "void"
		for declared in (_declaration(part),)
	}


def _top_level_parts(text: str) -> tuple[str, ...]:
	"""The comma-separated parts of a list, not splitting inside parentheses.

	>>> _top_level_parts("int a, void (*) (int, char) b, char c")
	('int a', 'void (*) (int, char) b', 'char c')
	"""
	depths = tuple(accumulate((character == "(") - (character == ")") for character in text))
	cuts = (
		-1,
		*(index for index, character in enumerate(text) if character == "," and depths[index] == 0),
		len(text),
	)
	return tuple(text[start + 1 : end].strip() for start, end in pairwise(cuts))


def _declaration(text: str) -> _Declared:
	"""A declared name and its type: GCC prints the name last, an array's bounds after it.

	>>> _declaration("  int (*<T9f7>) (int) _char_out.0_1;")
	_Declared(name='_char_out.0_1', type='int (*<T9f7>) (int)')
	>>> _declaration("  struct api local[2][3];")
	_Declared(name='local', type='struct api[2][3]')
	"""
	type_text, _, declarator = text.strip().removesuffix(";").rpartition(" ")
	name, bracket, bounds = declarator.partition("[")
	return _Declared(name=name, type=f"{type_text}{bracket}{bounds}")


def _statement(line: str) -> Statement | None:
	"""A statement, at ``<unknown>:0:0`` where GCC gave it no location, as the plugin writes it."""
	located = _LOCATION.match(line)
	body = _CALL_FLAGS.sub("", _LOCATION.sub("", line)).replace(" ={v} ", " = ")
	if not body or body.startswith("#") or _LABEL.match(body) is not None:
		return None
	return (
		Statement(
			file=located.group(1),
			line=int(located.group(2)),
			column=int(located.group(3)),
			body=body,
		)
		if located is not None
		else Statement(file="<unknown>", line=0, column=0, body=body)
	)
