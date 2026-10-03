# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Which of the image's functions a ``.ci`` node or ``.su`` record names."""

from __future__ import annotations

from itertools import groupby
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
	"""Each declaration exactly one function has."""
	by_symbol: Mapping[str, Address]
	"""Each symbol name exactly one function has."""
	names: Mapping[Address, str]
	"""Each function's name, unless another function's name has the same frame key."""


def identities(program: Program) -> Identities:
	starts = {
		normalized(address, program.machine): function.name
		for address, function in program.functions.items()
	}
	collided = frozenset(
		key
		for key, group in groupby(sorted(map(frame_key, starts.values())))
		if len(tuple(group)) > 1
	)
	return Identities(
		by_declaration={
			declaration: next(iter(addresses))
			for declaration, addresses in _addresses_by_declaration(program).items()
			if len(addresses) == 1
		},
		by_symbol={
			name: next(iter(addresses))
			for name, addresses in program.symbol_addresses.items()
			if len(addresses) == 1
		},
		names={start: name for start, name in starts.items() if frame_key(name) not in collided},
	)


def canonical_edges(
	known: Identities,
	unit: str,
	locations: Mapping[str, SourceLocation],
	edges: tuple[CallEdge, ...],
) -> tuple[CallEdge, ...]:
	return tuple(
		CallEdge(
			caller=_canonical(known, unit, locations, edge.caller),
			callee=_canonical(known, unit, locations, edge.callee),
			kind=edge.kind,
		)
		for edge in edges
	)


def canonical_frames(
	known: Identities,
	unit: str,
	locations: Mapping[str, SourceLocation],
	usages: tuple[StackUsage, ...],
) -> tuple[StackUsage, ...]:
	return tuple(
		StackUsage(
			function=_canonical(known, unit, locations, usage.function),
			bytes=usage.bytes,
			bounded=usage.bounded,
		)
		for usage in usages
	)


def _canonical(
	known: Identities, unit: str, locations: Mapping[str, SourceLocation], name: str
) -> str:
	match (
		known.by_declaration.get(Declaration(unit=unit, location=location))
		if (location := locations.get(name)) is not None
		else known.by_symbol.get(name)
	):
		case None:
			return name
		case address:
			return known.names.get(address, name)


def _addresses_by_declaration(program: Program) -> Mapping[Declaration, frozenset[Address]]:
	return {
		declaration: frozenset(address for _, address in group)
		for declaration, group in groupby(
			sorted(
				((declaration, address) for address, declaration in program.declarations.items()),
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
