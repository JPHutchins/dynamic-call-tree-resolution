# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Which of the image's functions a ``.ci`` node or ``.su`` record names."""

from __future__ import annotations

import posixpath
from functools import partial
from itertools import groupby, product
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from salix import Struct

from dynamic_call_tree_resolution.callgraph import CallEdge
from dynamic_call_tree_resolution.composition import flatmap, merged
from dynamic_call_tree_resolution.model import Address, Declaration
from dynamic_call_tree_resolution.stack_analysis import frame_key
from dynamic_call_tree_resolution.stack_usage import StackUsage
from dynamic_call_tree_resolution.vsa.abi import normalized

if TYPE_CHECKING:
	from collections.abc import Callable, Iterator, Mapping

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


class _KeyedName(Struct, order=True):
	"""A function's stack name under its frame key."""

	key: str
	name: str


class _KeyedStart(Struct, order=True):
	"""A function's start and symbol name under its frame key."""

	key: str
	start: Address
	name: str


class _DeclaredAt(Struct):
	"""A function's start, under the declaration it has."""

	declaration: Declaration
	address: Address


def identities(program: Program, names: Mapping[Address, str]) -> Identities:
	keyed_names = sorted(
		_KeyedName(key=frame_key(function.name), name=names[normalized(address, program.machine)])
		for address, function in program.functions.items()
		if normalized(address, program.machine) in names
	)
	names_by_key = {
		key: tuple(sorted(copy.name for copy in group))
		for key, group in groupby(keyed_names, key=_copy_key)
	}
	return Identities(
		by_declaration={
			declaration: next(iter(addresses))
			for declaration, addresses in _addresses_by_declaration(program).items()
			if len(addresses) == 1
		},
		by_symbol=program.symbol_addresses,
		names=names,
		copies={key: copies for key, copies in names_by_key.items() if len(copies) > 1},
	)


def stack_name(program: Program, names: Mapping[Address, str] | None, address: Address) -> str:
	return (
		names.get(normalized(address, program.machine), program.functions[address].name)
		if names is not None
		else program.functions[address].name
	)


def stack_names(program: Program) -> Mapping[Address, str]:
	keyed_starts = sorted(
		_KeyedStart(
			key=frame_key(function.name),
			start=normalized(address, program.machine),
			name=function.name,
		)
		for address, function in program.functions.items()
	)
	return merged(
		_distinct_names(tuple(group), program.declarations)
		for _, group in groupby(keyed_starts, key=_key)
	)


def canonical_edges(
	known: Identities,
	unit: str,
	locations: Mapping[str, SourceLocation],
	edges: tuple[CallEdge, ...],
) -> tuple[CallEdge, ...]:
	named = partial(_named, known, unit, locations)
	return tuple(flatmap(partial(_named_edges, named), edges))


def canonical_frames(
	known: Identities,
	unit: str,
	locations: Mapping[str, SourceLocation],
	usages: tuple[StackUsage, ...],
) -> tuple[StackUsage, ...]:
	named = partial(_named, known, unit, locations)
	return tuple(flatmap(partial(_named_frames, named), usages))


def _named_edges(named: Callable[[str], tuple[str, ...]], edge: CallEdge) -> Iterator[CallEdge]:
	return (
		CallEdge(caller=caller, callee=callee, kind=edge.kind)
		for caller, callee in product(named(edge.caller), named(edge.callee))
	)


def _named_frames(
	named: Callable[[str], tuple[str, ...]], usage: StackUsage
) -> Iterator[StackUsage]:
	return (
		StackUsage(function=function, bytes=usage.bytes, bounded=usage.bounded)
		for function in named(usage.function)
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
	group: tuple[_KeyedStart, ...], declarations: Mapping[Address, Declaration]
) -> Mapping[Address, str]:
	match group:
		case (only,):
			return {only.start: only.name}
		case _:
			return _qualified(group, declarations)


def _qualified(
	group: tuple[_KeyedStart, ...], declarations: Mapping[Address, Declaration]
) -> Mapping[Address, str]:
	paths = {
		keyed.start: PurePosixPath(posixpath.normpath(declarations[keyed.start].unit)).parts
		if keyed.start in declarations
		else ()
		for keyed in group
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
		keyed.start: f"{keyed.key}@{'/'.join(paths[keyed.start][-depth:])}"
		if depth is not None
		else f"{keyed.key}@{keyed.start:#x}"
		for keyed in group
	}


def _copy_key(copy: _KeyedName) -> str:
	return copy.key


def _key(keyed: _KeyedStart) -> str:
	return keyed.key


def _addresses_by_declaration(program: Program) -> Mapping[Declaration, frozenset[Address]]:
	return {
		declaration: frozenset(declared.address for declared in group)
		for declaration, group in groupby(
			sorted(
				(
					_DeclaredAt(
						declaration=Declaration(
							unit=PurePosixPath(declaration.unit).name, location=declaration.location
						),
						address=address,
					)
					for address, declaration in program.declarations.items()
				),
				key=_declared,
			),
			key=_declared,
		)
	}


def _declared(declared: _DeclaredAt) -> Declaration:
	return declared.declaration
