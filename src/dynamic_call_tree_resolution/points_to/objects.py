# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The data objects and DWARF layouts that hold an image's slots."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from dynamic_call_tree_resolution.model import (
		Address,
		DataObject,
		Member,
		Program,
		StructureLayout,
	)


def slot_path(program: Program, address: Address) -> tuple[str | None, ...]:
	owner = object_covering(program, address)
	if owner is None:
		return ()  # pragma: no cover
	layout = layout_of(program, owner)
	if layout is None:
		return (owner.name,)
	member = _member_at(layout, address - owner.address)
	return (owner.name, member.name) if member is not None else (owner.name,)  # pragma: no branch


def object_covering(program: Program, address: Address | None) -> DataObject | None:
	if address is None:
		return None
	return next(
		(
			data_object
			for data_object in program.objects.values()
			if data_object.address <= address < data_object.address + max(data_object.size, 1)
		),
		None,
	)


def layout_of(program: Program, data_object: DataObject) -> StructureLayout | None:
	if data_object.type_name is None:
		return None
	return program.layouts.get(data_object.type_name)


def _member_at(layout: StructureLayout, offset: int) -> Member | None:
	return next(
		(member for member in layout.members if member.offset == offset), None
	)  # pragma: no branch
