# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""What the references the linker kept say about the image's functions."""

from __future__ import annotations

from functools import partial
from itertools import chain, groupby
from typing import TYPE_CHECKING

from salix import Struct

from dynamic_call_tree_resolution.model import Address, ReferenceKind
from dynamic_call_tree_resolution.vsa.abi import normalized

if TYPE_CHECKING:
	from collections.abc import Iterator, Mapping

	from dynamic_call_tree_resolution.model import Function, Machine, Program


class LinkedCall(Struct):
	"""A direct call or tail call the linker resolved."""

	slot: Address
	caller: Address | None
	"""The function whose code holds the call; none outside every function."""
	callee: Address


def linked_calls(program: Program) -> frozenset[LinkedCall]:
	return frozenset(
		LinkedCall(
			slot=slot,
			caller=caller.address if caller is not None else None,
			callee=callee,
		)
		for slot, callee in linked_targets(program, ReferenceKind.CALL)
		for caller in (function_covering(program, slot),)
	)


def linked_targets(program: Program, kind: ReferenceKind) -> Iterator[tuple[Address, Address]]:
	starts = {normalized(address, program.machine): address for address in program.functions}
	return (
		(reference.slot, function)
		for reference in program.link_references
		if reference.kind is kind
		and reference.symbol
		and (
			function := reference.value
			if reference.value in program.functions
			else starts.get(normalized(reference.value, program.machine))
		)
		is not None
	)


def function_covering(program: Program, address: Address) -> Function | None:
	return next(
		chain(
			(
				function
				for function in program.functions.values()
				if address - normalized(function.address, program.machine) in range(function.size)
			),
			_assembly_covering(program, address),
		),
		None,
	)


def _assembly_covering(program: Program, address: Address) -> Iterator[Function]:
	yield from (
		function
		for functions in (functions_by_start(program),)
		for start, function in functions.items()
		if function.size == 0
		and address - start in range(function_extent(program, functions, start))
	)


def function_extent(program: Program, functions: Mapping[Address, Function], start: Address) -> int:
	return (
		functions[start].size
		or min(
			(
				*(following for following in functions if following > start),
				*(
					base + len(section.data)
					for base, section in program.sections.items()
					if base <= start < base + len(section.data)
				),
			),
			default=start,
		)
		- start
	)


def functions_by_start(program: Program) -> Mapping[Address, Function]:
	return {
		start: max(group, key=_size)
		for start, group in groupby(
			sorted(program.functions.values(), key=partial(_code_start, program.machine)),
			key=partial(_code_start, program.machine),
		)
	}


def _code_start(machine: Machine, function: Function) -> Address:
	return normalized(function.address, machine)


def _size(function: Function) -> int:
	return function.size
