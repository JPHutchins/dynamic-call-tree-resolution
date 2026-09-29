# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Shared hand-built Program fixtures."""

from typing import TYPE_CHECKING

from dynamic_call_tree_resolution import Address, DataObject, Function, Machine, Program

if TYPE_CHECKING:
	from collections.abc import Mapping


def build_program(
	machine: str,
	functions: tuple[tuple[str, int, int], ...] = (),
	*,
	objects: tuple[tuple[str, int, bytes], ...] = (),
	sections: Mapping[int, bytes] | None = None,
	pointer_size: int = 8,
) -> Program:
	return Program(
		byte_order="little",
		pointer_size=pointer_size,
		machine=Machine(machine),
		functions={
			Address(address): Function(
				name=name, address=Address(address), size=size, signature=None
			)
			for name, address, size in functions
		},
		objects={
			Address(address): DataObject(
				name=name,
				address=Address(address),
				size=len(data),
				type_name=None,
				signature=None,
			)
			for name, address, data in objects
		},
		layouts={},
		relocations=(),
		sections={
			Address(address): data
			for address, data in (sections if sections is not None else dict[int, bytes]()).items()
		}
		| {Address(address): data for _, address, data in objects},
	)
