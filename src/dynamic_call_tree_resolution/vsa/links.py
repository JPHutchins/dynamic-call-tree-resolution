# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""What the references the linker kept say about the image's functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from salix import Struct

from dynamic_call_tree_resolution.model import Address, ReferenceKind
from dynamic_call_tree_resolution.vsa.abi import normalized

if TYPE_CHECKING:
	from collections.abc import Iterator

	from dynamic_call_tree_resolution.model import Function, Program


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
		(
			function
			for function in program.functions.values()
			if address - normalized(function.address, program.machine) in range(function.size)
		),
		None,
	)
