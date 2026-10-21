# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The targets the image's initializers and relocations store in each slot."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from salix import Struct

from dynamic_call_tree_resolution.model import (
	ARRAY_SUFFIX,
	FUNCTION_POINTER,
	Address,
	ArrayMember,
	EmbeddedStructMember,
	FunctionPointerMember,
	Program,
	Provenance,
	SkippedMember,
	SlotAssignment,
	StructPointerMember,
	array_element_type,
	array_elements,
)
from dynamic_call_tree_resolution.points_to.image import in_writable_memory, pointer_at
from dynamic_call_tree_resolution.points_to.objects import layout_of, object_covering, slot_path

if TYPE_CHECKING:
	from collections.abc import Iterable

	from dynamic_call_tree_resolution.model import DataObject, Member


class _Resolution(Struct):
	"""A slot and the targets one walk of the image found for it."""

	slot: Address
	path: tuple[str | None, ...]
	candidates: frozenset[Address]


def assignments(program: Program) -> tuple[SlotAssignment, ...]:
	return tuple(
		sorted(
			(
				SlotAssignment(
					slot=resolution.slot,
					path=resolution.path,
					candidates=resolution.candidates,
					provenance=_provenance(program, resolution.slot),
					relocated=relocated,
				)
				for resolution, relocated in {
					**dict.fromkeys(_from_objects(program), False),
					**dict.fromkeys(_from_relocations(program), True),
				}.items()
			),
			key=_assignment_slot,
		)
	)


def _provenance(program: Program, slot: Address) -> Provenance:
	return (
		Provenance.RAM_INITIALIZER if in_writable_memory(program, slot) else Provenance.ROM_CONSTANT
	)


def _from_relocations(program: Program) -> Iterable[_Resolution]:
	for relocation in program.relocations:
		if relocation.target not in program.functions:
			continue
		yield _Resolution(
			slot=relocation.slot,
			path=slot_path(program, relocation.slot),
			candidates=frozenset({relocation.target}),
		)


def _from_objects(program: Program) -> Iterable[_Resolution]:
	for data_object in program.objects.values():
		yield from _object_assignments(
			program, data_object, path=(data_object.name,), visited=frozenset()
		)


def _object_assignments(
	program: Program,
	data_object: DataObject,
	path: tuple[str | None, ...],
	visited: frozenset[Address],
) -> Iterable[_Resolution]:
	if data_object.address in visited:
		return
	layout = layout_of(program, data_object)
	if layout is None:
		if data_object.type_name is not None and data_object.type_name.endswith(ARRAY_SUFFIX):
			element_name = array_element_type(data_object.type_name)
			if program.layouts.get(element_name) is not None or element_name == FUNCTION_POINTER:
				yield from _array_element_assignments(program, data_object, path, visited)
			else:
				yield from _vector_table_assignments(program, data_object, path)
		elif data_object.size > program.pointer_size:
			yield from _vector_table_assignments(program, data_object, path)
		else:
			target = pointer_at(program, data_object.address)
			if target is not None and target != 0 and target in program.functions:
				yield _Resolution(
					slot=data_object.address,
					path=path,
					candidates=frozenset({target}),
				)
		return
	visited = visited | {data_object.address}
	yield from _member_assignments(program, data_object.address, 0, layout.members, path, visited)


def _vector_table_assignments(
	program: Program,
	data_object: DataObject,
	path: tuple[str | None, ...],
) -> Iterable[_Resolution]:
	values = tuple(
		pointer_at(program, Address(data_object.address + base_offset))
		for base_offset in range(0, data_object.size, program.pointer_size)
	)
	targets = tuple(
		target
		for target in values
		if target is not None and target != 0 and target in program.functions
	)
	if len(targets) != len(values):
		return
	for index, target in enumerate(targets):
		yield _Resolution(
			slot=Address(data_object.address + index * program.pointer_size),
			path=(*path, f"[{index}]"),
			candidates=frozenset({target}),
		)


def _array_element_assignments(
	program: Program,
	data_object: DataObject,
	path: tuple[str | None, ...],
	visited: frozenset[Address],
) -> Iterable[_Resolution]:
	element_name = array_element_type(data_object.type_name) if data_object.type_name else ""
	element_layout = program.layouts.get(element_name)
	if element_layout is None:
		for index, base_offset in enumerate(range(0, data_object.size, program.pointer_size)):
			target = pointer_at(program, Address(data_object.address + base_offset))
			if target is not None and target != 0 and target in program.functions:
				yield _Resolution(
					slot=Address(data_object.address + base_offset),
					path=(*path, f"[{index}]"),
					candidates=frozenset({target}),
				)
		return
	for index, base_offset in enumerate(range(0, data_object.size, element_layout.size or 1)):
		yield from _member_assignments(
			program,
			data_object.address,
			base_offset,
			element_layout.members,
			(*path, f"[{index}]"),
			visited,
		)


def _member_assignments(
	program: Program,
	base_address: Address,
	base_offset: int,
	members: tuple[Member, ...],
	path: tuple[str | None, ...],
	visited: frozenset[Address],
) -> Iterable[_Resolution]:
	for member in members:
		member_path = (*path, member.name)
		match member:
			case FunctionPointerMember(offset=offset):
				target = pointer_at(program, Address(base_address + base_offset + offset))
				if target is not None and target != 0 and target in program.functions:
					yield _Resolution(
						slot=Address(base_address + base_offset + offset),
						path=member_path,
						candidates=frozenset({target}),
					)
			case StructPointerMember(offset=offset, pointee=pointee):
				target_object = object_covering(
					program,
					pointer_at(program, Address(base_address + base_offset + offset)) or None,
				)
				if target_object is not None and (
					pointee is None
					or target_object.type_name is None
					or target_object.type_name == pointee
				):
					yield from _object_assignments(program, target_object, member_path, visited)
			case EmbeddedStructMember(offset=offset, members=inner_members):
				yield from _member_assignments(
					program,
					base_address,
					base_offset + offset,
					inner_members,
					member_path,
					visited,
				)
			case ArrayMember(offset=offset):
				yield from _member_assignments(
					program,
					base_address,
					base_offset + offset,
					array_elements(member),
					member_path,
					visited,
				)
			case SkippedMember():
				pass
			case _ as unreachable:
				assert_never(unreachable)


def _assignment_slot(assignment: SlotAssignment) -> Address:
	return assignment.slot
