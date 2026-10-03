# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Which of the image's functions a ``.ci`` node or ``.su`` record names."""

from __future__ import annotations

import posixpath
from itertools import groupby
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from salix import Struct

from dynamic_call_tree_resolution.callgraph import CallEdge
from dynamic_call_tree_resolution.model import Address, Declaration
from dynamic_call_tree_resolution.stack_analysis import frame_key
from dynamic_call_tree_resolution.stack_usage import StackUsage
from dynamic_call_tree_resolution.vsa.abi import normalized

if TYPE_CHECKING:
	from collections.abc import Mapping

	from dynamic_call_tree_resolution.model import Program, SourceLocation


class Identities(Struct):
	"""The image's functions, found by declaration or by symbol name."""

	by_declaration: Mapping[Declaration, Address]
	"""Each declaration exactly one function has, by its unit's file name."""
	by_symbol: Mapping[str, frozenset[Address]]
	names: Mapping[Address, str]
	"""Each function's name in the stack graph, qualified by its unit where names collide."""
	copies: Mapping[str, tuple[str, ...]]
	"""The names of the functions that share each colliding frame key."""


def identities(program: Program, names: Mapping[Address, str]) -> Identities:
	return Identities(
		by_declaration={
			declaration: next(iter(addresses))
			for declaration, addresses in _addresses_by_declaration(program).items()
			if len(addresses) == 1
		},
		by_symbol=program.symbol_addresses,
		names=names,
		copies={
			key: tuple(sorted(name for _, name in copies))
			for key, group in groupby(
				sorted(
					(frame_key(function.name), names[normalized(address, program.machine)])
					for address, function in program.functions.items()
					if normalized(address, program.machine) in names
				),
				key=_copy_key,
			)
			for copies in (tuple(group),)
			if len(copies) > 1
		},
	)


def stack_name(program: Program, names: Mapping[Address, str] | None, address: Address) -> str:
	return (
		names.get(normalized(address, program.machine), program.functions[address].name)
		if names is not None
		else program.functions[address].name
	)


def stack_names(program: Program) -> Mapping[Address, str]:
	return {
		start: name
		for _, group in groupby(
			sorted(
				(
					(frame_key(function.name), normalized(address, program.machine), function.name)
					for address, function in program.functions.items()
				),
			),
			key=_key,
		)
		for start, name in _distinct_names(tuple(group), program.declarations).items()
	}


def canonical_edges(
	known: Identities,
	unit: str,
	locations: Mapping[str, SourceLocation],
	edges: tuple[CallEdge, ...],
) -> tuple[CallEdge, ...]:
	return tuple(
		CallEdge(caller=caller, callee=callee, kind=edge.kind)
		for edge in edges
		for caller in _named(known, unit, locations, edge.caller)
		for callee in _named(known, unit, locations, edge.callee)
	)


def canonical_frames(
	known: Identities,
	unit: str,
	locations: Mapping[str, SourceLocation],
	usages: tuple[StackUsage, ...],
) -> tuple[StackUsage, ...]:
	return tuple(
		StackUsage(function=function, bytes=usage.bytes, bounded=usage.bounded)
		for usage in usages
		for function in _named(known, unit, locations, usage.function)
	)


def _named(
	known: Identities, unit: str, locations: Mapping[str, SourceLocation], name: str
) -> tuple[str, ...]:
	match (
		known.by_declaration.get(Declaration(unit=unit, location=location))
		if (location := locations.get(name)) is not None
		else None
	):
		case None:
			return tuple(
				sorted(
					known.names[address]
					for address in known.by_symbol.get(name, frozenset())
					if address in known.names
				)
			) or known.copies.get(frame_key(name), (name,))
		case address:
			return (known.names.get(address, name),)


def _distinct_names(
	group: tuple[tuple[str, Address, str], ...], declarations: Mapping[Address, Declaration]
) -> Mapping[Address, str]:
	match group:
		case ((_, start, name),):
			return {start: name}
		case _:
			return _qualified(group, declarations)


def _qualified(
	group: tuple[tuple[str, Address, str], ...], declarations: Mapping[Address, Declaration]
) -> Mapping[Address, str]:
	paths = {
		start: PurePosixPath(posixpath.normpath(declarations[start].unit)).parts
		if start in declarations
		else ()
		for _, start, _ in group
	}
	depth = next(
		(
			depth
			for depth in range(1, max(map(len, paths.values()), default=0) + 1)
			if all(paths.values())
			and len({parts[-depth:] for parts in paths.values()}) == len(group)
		),
		None,
	)
	return {
		start: f"{key}@{'/'.join(paths[start][-depth:])}"
		if depth is not None
		else f"{key}@{start:#x}"
		for key, start, _ in group
	}


def _copy_key(entry: tuple[str, str]) -> str:
	return entry[0]


def _key(entry: tuple[str, Address, str]) -> str:
	return entry[0]


def _addresses_by_declaration(program: Program) -> Mapping[Declaration, frozenset[Address]]:
	return {
		declaration: frozenset(address for _, address in group)
		for declaration, group in groupby(
			sorted(
				(
					(
						Declaration(
							unit=PurePosixPath(declaration.unit).name, location=declaration.location
						),
						address,
					)
					for address, declaration in program.declarations.items()
				),
				key=_declaration_order,
			),
			key=_declared,
		)
	}


def _declaration_order(entry: tuple[Declaration, Address]) -> tuple[str, str, int, int]:
	declaration, _ = entry
	return (
		declaration.unit,
		declaration.location.file,
		declaration.location.line,
		declaration.location.column,
	)


def _declared(entry: tuple[Declaration, Address]) -> Declaration:
	return entry[0]
