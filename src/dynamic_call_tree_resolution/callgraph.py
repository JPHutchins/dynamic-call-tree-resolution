# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Parsing GCC ``-fcallgraph-info`` (``.ci`` VCG) call graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from salix import Struct

if TYPE_CHECKING:
	from pathlib import Path


class CallEdge(Struct):
	"""A resolved call edge from a ``.ci`` file."""

	caller: str
	callee: str


class _VcgEdge(Struct):
	"""One raw ``.ci`` edge entry."""

	sourcename: str
	targetname: str


def parse_callgraph(path: Path) -> tuple[CallEdge, ...]:
	return tuple(
		CallEdge(caller=edge.sourcename, callee=edge.targetname)
		for edge in _parse_vcg(path.read_text())
	)


def load_callgraph(build_directory: Path) -> tuple[CallEdge, ...]:
	return tuple(
		edge
		for callgraph_file in sorted(build_directory.glob("**/*.ci"))
		for edge in parse_callgraph(callgraph_file)
	)


def _parse_vcg(text: str) -> tuple[_VcgEdge, ...]:
	tokens = _tokenize(text)
	entries, _ = _parse_body(tokens, 0, "graph")
	return tuple(_edge(entry) for entry in entries.get("edge", []))


def _edge(entry: dict[str, str]) -> _VcgEdge:
	missing = next((key for key in ("sourcename", "targetname") if key not in entry), None)
	if missing is not None:
		raise ValueError(f"edge entry is missing {missing!r}")
	return _VcgEdge(sourcename=entry["sourcename"], targetname=entry["targetname"])


def _tokenize(text: str) -> tuple[str, ...]:
	tokens: list[str] = []
	index = 0
	while index < len(text):
		if text[index].isspace():
			index += 1
			continue
		if text[index] in "{}:":
			tokens.append(text[index])
			index += 1
			continue
		if text[index] == '"':
			end = text.index('"', index + 1)
			tokens.append(text[index + 1 : end])
			index = end + 1
			continue
		start = index
		while index < len(text) and not text[index].isspace() and text[index] not in "{}:":
			index += 1
		tokens.append(text[start:index])
	return tuple(tokens)


def _parse_body(
	tokens: tuple[str, ...],
	index: int,
	key: str,
) -> tuple[dict[str, list[dict[str, str]]], int]:
	if tokens[index] != key or tokens[index + 1] != ":" or tokens[index + 2] != "{":
		raise ValueError(f"expected '{key}: {{' at token {index}")
	entries: dict[str, list[dict[str, str]]] = {}
	index += 3
	while tokens[index] != "}":
		entry_key = tokens[index]
		if tokens[index + 1] != ":":
			raise ValueError(f"expected ':' after '{entry_key}' at token {index + 1}")
		index += 2
		if tokens[index] == "{":
			entry, index = _parse_entry(tokens, index)
			entries.setdefault(entry_key, []).append(entry)
		else:
			index += 1
	return entries, index + 1


def _parse_entry(tokens: tuple[str, ...], index: int) -> tuple[dict[str, str], int]:
	entry: dict[str, str] = {}
	index += 1
	while tokens[index] != "}":
		entry_key = tokens[index]
		if tokens[index + 1] != ":":
			raise ValueError(f"expected ':' after '{entry_key}' at token {index + 1}")
		index += 2
		entry[entry_key] = tokens[index]
		index += 1
	return entry, index + 1
