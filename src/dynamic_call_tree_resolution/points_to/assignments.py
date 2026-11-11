# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The targets the image's initializers and relocations store in each slot."""

from __future__ import annotations

from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, assert_never

from salix import Struct

from dynamic_call_tree_resolution.composition import flatmap
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
	return (
		_Resolution(
			slot=relocation.slot,
			path=slot_path(program, relocation.slot),
			candidates=frozenset({relocation.target}),
		)
		for relocation in program.relocations
		if relocation.target in program.functions
	)


def _from_objects(program: Program) -> Iterable[_Resolution]:
	return flatmap(
		lambda data_object: _object_assignments(
			program, data_object, path=(data_object.name,), visited=frozenset()
		),
		program.objects.values(),
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
			yield from _slot_resolutions(program, data_object.address, path)
		return
	yield from _member_assignments(
		program, data_object.address, 0, layout.members, path, visited | {data_object.address}
	)


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
		return ()
	return (
		_Resolution(
			slot=Address(data_object.address + index * program.pointer_size),
			path=(*path, f"[{index}]"),
			candidates=frozenset({target}),
		)
		for index, target in enumerate(targets)
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
		return chain.from_iterable(
			_slot_resolutions(
				program, Address(data_object.address + base_offset), (*path, f"[{index}]")
			)
			for index, base_offset in enumerate(range(0, data_object.size, program.pointer_size))
		)
	return chain.from_iterable(
		_member_assignments(
			program,
			data_object.address,
			base_offset,
			element_layout.members,
			(*path, f"[{index}]"),
			visited,
		)
		for index, base_offset in enumerate(range(0, data_object.size, element_layout.size or 1))
	)


def _member_assignments(
	program: Program,
	base_address: Address,
	base_offset: int,
	members: tuple[Member, ...],
	path: tuple[str | None, ...],
	visited: frozenset[Address],
) -> Iterable[_Resolution]:
	return flatmap(
		partial(_assignments_in_member, program, base_address, base_offset, path, visited), members
	)


def _assignments_in_member(
	program: Program,
	base_address: Address,
	base_offset: int,
	path: tuple[str | None, ...],
	visited: frozenset[Address],
	member: Member,
) -> Iterable[_Resolution]:
	member_path = (*path, member.name)
	match member:
		case FunctionPointerMember(offset=offset):
			return _slot_resolutions(
				program, Address(base_address + base_offset + offset), member_path
			)
		case StructPointerMember(offset=offset, pointee=pointee):
			target_object = object_covering(
				program,
				pointer_at(program, Address(base_address + base_offset + offset)) or None,
			)
			return (
				_object_assignments(program, target_object, member_path, visited)
				if target_object is not None
				and (
					pointee is None
					or target_object.type_name is None
					or target_object.type_name == pointee
				)
				else ()
			)
		case EmbeddedStructMember(offset=offset, members=inner_members):
			return _member_assignments(
				program,
				base_address,
				base_offset + offset,
				inner_members,
				member_path,
				visited,
			)
		case ArrayMember(offset=offset):
			return _member_assignments(
				program,
				base_address,
				base_offset + offset,
				array_elements(member),
				member_path,
				visited,
			)
		case SkippedMember():
			return ()
		case _ as unreachable:
			assert_never(unreachable)


def _slot_resolutions(
	program: Program, slot: Address, path: tuple[str | None, ...]
) -> tuple[_Resolution, ...]:
	target = pointer_at(program, slot)
	return (
		(_Resolution(slot=slot, path=path, candidates=frozenset({target})),)
		if target is not None and target != 0 and target in program.functions
		else ()
	)


def _assignment_slot(assignment: SlotAssignment) -> Address:
	return assignment.slot
