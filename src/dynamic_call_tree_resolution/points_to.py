# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Static resolution of function-pointer slots to candidate target functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dynamic_call_tree_resolution.model import (
	Address,
	EmbeddedStructMember,
	FunctionPointerMember,
	Program,
	Provenance,
	SlotAssignment,
	StructPointerMember,
)

if TYPE_CHECKING:
	from collections.abc import Iterable

	from dynamic_call_tree_resolution.model import DataObject, Member, StructureLayout


def assignments(program: Program) -> tuple[SlotAssignment, ...]:
	"""Resolve every statically-known function-pointer slot in ``program``.

	Covers direct function-pointer globals (via relocations and baked
	constant data) and slots reached through constant struct-pointer
	chains, such as Zephyr ``device->api->fn`` structures.
	"""
	by_key: dict[tuple[Address, tuple[str | None, ...], frozenset[Address]], Provenance] = {}
	for assignment in _from_objects(program):
		by_key[(assignment.slot, assignment.path, assignment.candidates)] = assignment.provenance
	for assignment in _from_relocations(program):
		by_key[(assignment.slot, assignment.path, assignment.candidates)] = assignment.provenance
	return tuple(
		sorted(
			(
				SlotAssignment(
					slot=slot,
					path=path,
					candidates=candidates,
					provenance=provenance,
				)
				for (slot, path, candidates), provenance in by_key.items()
			),
			key=lambda assignment: assignment.slot,
		)
	)


def _from_relocations(program: Program) -> Iterable[SlotAssignment]:
	for relocation in program.relocations:
		if relocation.target not in program.functions:
			continue
		yield SlotAssignment(
			slot=relocation.slot,
			path=_slot_path(program, relocation.slot),
			candidates=frozenset({relocation.target}),
			provenance=Provenance.RELOCATION,
		)


def _from_objects(program: Program) -> Iterable[SlotAssignment]:
	for data_object in program.objects.values():
		yield from _object_assignments(
			program, data_object, path=(data_object.name,), visited=frozenset()
		)


def _object_assignments(
	program: Program,
	data_object: DataObject,
	path: tuple[str | None, ...],
	visited: frozenset[Address],
) -> Iterable[SlotAssignment]:
	if data_object.address in visited:
		return
	layout = _layout_of(program, data_object)
	if layout is None:
		target = _read_pointer(program, data_object, 0)
		if target is not None and target in program.functions:
			yield SlotAssignment(
				slot=data_object.address,
				path=path,
				candidates=frozenset({target}),
				provenance=Provenance.CONSTANT_DATA,
			)
		return
	visited = visited | {data_object.address}
	yield from _member_assignments(program, data_object, 0, layout.members, path, visited)


def _member_assignments(
	program: Program,
	data_object: DataObject,
	base_offset: int,
	members: tuple[Member, ...],
	path: tuple[str | None, ...],
	visited: frozenset[Address],
) -> Iterable[SlotAssignment]:
	for member in members:
		member_path = (*path, member.name)
		match member:
			case FunctionPointerMember(offset=offset):
				target = _read_pointer(program, data_object, base_offset + offset)
				if target is not None and target in program.functions:
					yield SlotAssignment(
						slot=Address(data_object.address + base_offset + offset),
						path=member_path,
						candidates=frozenset({target}),
						provenance=Provenance.CONSTANT_DATA,
					)
			case StructPointerMember(offset=offset, pointee=pointee):  # pragma: no branch
				target_object = _object_covering(
					program, _read_pointer(program, data_object, base_offset + offset)
				)
				if target_object is not None and (
					pointee is None
					or target_object.type_name is None
					or target_object.type_name == pointee
				):
					yield from _object_assignments(program, target_object, member_path, visited)
			case EmbeddedStructMember(offset=offset, members=inner_members):  # pragma: no branch
				yield from _member_assignments(
					program,
					data_object,
					base_offset + offset,
					inner_members,
					member_path,
					visited,
				)


def _slot_path(program: Program, address: Address) -> tuple[str | None, ...]:
	owner = _object_covering(program, address)
	if owner is None:
		return ()  # pragma: no cover
	layout = _layout_of(program, owner)
	if layout is None:
		return (owner.name,)
	member = _member_at(layout, address - owner.address)
	return (owner.name, member.name) if member is not None else (owner.name,)  # pragma: no branch


def _layout_of(program: Program, data_object: DataObject) -> StructureLayout | None:
	if data_object.type_name is None:
		return None
	return program.layouts.get(data_object.type_name)


def _member_at(layout: StructureLayout, offset: int) -> Member | None:
	return next(
		(member for member in layout.members if member.offset == offset), None
	)  # pragma: no branch


def _object_covering(program: Program, address: Address | None) -> DataObject | None:
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


def _read_pointer(program: Program, data_object: DataObject, offset: int) -> Address | None:
	if offset + program.pointer_size > len(data_object.bytes):
		return None
	return Address(
		int.from_bytes(
			data_object.bytes[offset : offset + program.pointer_size], program.byte_order
		)
	)
