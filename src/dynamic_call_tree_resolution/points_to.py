# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Static resolution of function-pointer slots to candidate target functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from salix import Struct

from dynamic_call_tree_resolution.model import (
	ARRAY_SUFFIX,
	FUNCTION_POINTER,
	Address,
	EmbeddedStructMember,
	FunctionPointerMember,
	FunctionSignature,
	Program,
	Provenance,
	SlotAssignment,
	StructPointerMember,
	UnresolvedSlot,
	array_element_type,
)

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping

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
			key=_assignment_slot,
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
				yield SlotAssignment(
					slot=data_object.address,
					path=path,
					candidates=frozenset({target}),
					provenance=Provenance.CONSTANT_DATA,
				)
		return
	visited = visited | {data_object.address}
	yield from _member_assignments(program, data_object.address, 0, layout.members, path, visited)


def _vector_table_assignments(
	program: Program,
	data_object: DataObject,
	path: tuple[str | None, ...],
) -> Iterable[SlotAssignment]:
	"""Typeless objects whose every element is a function: vector tables.

	Assembly-defined vector tables (``_irq_vector_table`` and friends)
	have no DWARF type; a typeless object whose elements are all known
	function addresses is one.
	"""
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
		yield SlotAssignment(
			slot=Address(data_object.address + index * program.pointer_size),
			path=(*path, f"[{index}]"),
			candidates=frozenset({target}),
			provenance=Provenance.CONSTANT_DATA,
		)


def _array_element_assignments(
	program: Program,
	data_object: DataObject,
	path: tuple[str | None, ...],
	visited: frozenset[Address],
) -> Iterable[SlotAssignment]:
	element_name = array_element_type(data_object.type_name) if data_object.type_name else ""
	element_layout = program.layouts.get(element_name)
	if element_layout is None:
		for index, base_offset in enumerate(range(0, data_object.size, program.pointer_size)):
			target = pointer_at(program, Address(data_object.address + base_offset))
			if target is not None and target != 0 and target in program.functions:
				yield SlotAssignment(
					slot=Address(data_object.address + base_offset),
					path=(*path, f"[{index}]"),
					candidates=frozenset({target}),
					provenance=Provenance.CONSTANT_DATA,
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
) -> Iterable[SlotAssignment]:
	for member in members:
		member_path = (*path, member.name)
		match member:
			case FunctionPointerMember(offset=offset):
				target = pointer_at(program, Address(base_address + base_offset + offset))
				if target is not None and target != 0 and target in program.functions:
					yield SlotAssignment(
						slot=Address(base_address + base_offset + offset),
						path=member_path,
						candidates=frozenset({target}),
						provenance=Provenance.CONSTANT_DATA,
					)
			case StructPointerMember(offset=offset, pointee=pointee):  # pragma: no branch
				target_object = _object_covering(
					program,
					pointer_at(program, Address(base_address + base_offset + offset)),
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
					base_address,
					base_offset + offset,
					inner_members,
					member_path,
					visited,
				)


class _SlotUniverseEntry(Struct):
	path: tuple[str | None, ...]
	signature: FunctionSignature | None


def _assignment_slot(assignment: SlotAssignment) -> Address:
	return assignment.slot


def _slot_address(slot: UnresolvedSlot) -> Address:
	return slot.slot


def unresolved_slots(
	program: Program, resolved: tuple[SlotAssignment, ...]
) -> tuple[UnresolvedSlot, ...]:
	"""Every function-pointer slot in the image with no resolved candidates.

	The slot universe covers top-level function-pointer globals, every
	function-pointer member of every object with a known structure layout
	(including embedded structs), and relocation slots pointing at
	functions.
	"""
	resolved_by_slot = {assignment.slot for assignment in resolved}
	universe: dict[Address, _SlotUniverseEntry] = {}
	for data_object in program.objects.values():
		layout = _layout_of(program, data_object)
		if layout is None:
			if data_object.type_name == FUNCTION_POINTER:
				universe.setdefault(
					data_object.address,
					_SlotUniverseEntry(path=(data_object.name,), signature=data_object.signature),
				)
			elif data_object.type_name is not None and data_object.type_name.endswith(ARRAY_SUFFIX):
				_array_element_slots(program, data_object, (data_object.name,), universe)
			continue
		_collect_member_slots(data_object, 0, layout.members, (data_object.name,), universe)
	for relocation in program.relocations:
		if relocation.target not in program.functions:
			continue
		universe.setdefault(
			relocation.slot,
			_SlotUniverseEntry(path=_slot_path(program, relocation.slot), signature=None),
		)
	return tuple(
		sorted(
			(
				UnresolvedSlot(slot=slot, path=entry.path, signature=entry.signature)
				for slot, entry in universe.items()
				if slot not in resolved_by_slot
			),
			key=_slot_address,
		)
	)


def _array_element_slots(
	program: Program,
	data_object: DataObject,
	path: tuple[str | None, ...],
	universe: dict[Address, _SlotUniverseEntry],
) -> None:
	element_name = array_element_type(data_object.type_name) if data_object.type_name else ""
	element_layout = program.layouts.get(element_name)
	if element_layout is None:
		if element_name == FUNCTION_POINTER:
			for index, base_offset in enumerate(range(0, data_object.size, program.pointer_size)):
				universe.setdefault(
					Address(data_object.address + base_offset),
					_SlotUniverseEntry(path=(*path, f"[{index}]"), signature=data_object.signature),
				)
		return
	for index, base_offset in enumerate(range(0, data_object.size, element_layout.size or 1)):
		_collect_member_slots(
			data_object,
			base_offset,
			element_layout.members,
			(*path, f"[{index}]"),
			universe,
		)


def _collect_member_slots(
	data_object: DataObject,
	base_offset: int,
	members: tuple[Member, ...],
	path: tuple[str | None, ...],
	universe: dict[Address, _SlotUniverseEntry],
) -> None:
	for member in members:
		member_path = (*path, member.name)
		match member:
			case FunctionPointerMember(offset=offset, signature=signature):
				universe.setdefault(
					Address(data_object.address + base_offset + offset),
					_SlotUniverseEntry(path=member_path, signature=signature),
				)
			case EmbeddedStructMember(offset=offset, members=inner_members):
				_collect_member_slots(
					data_object, base_offset + offset, inner_members, member_path, universe
				)
			case StructPointerMember():  # pragma: no branch
				pass


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


def signatures_by_slot(
	unresolved: tuple[UnresolvedSlot, ...],
) -> Mapping[Address, FunctionSignature]:
	"""Signatures of the unresolved slot universe, keyed by slot address."""
	return {slot.slot: slot.signature for slot in unresolved if slot.signature is not None}


def memory_at(program: Program, address: Address, size: int) -> bytes:
	"""Read ``size`` bytes of the loaded image at ``address``.

	Returns a short (possibly empty) slice when the read extends past the
	end of the covering allocated section.
	"""
	for section_address, data in program.sections.items():
		if not section_address <= address < section_address + len(data):
			continue
		offset = address - section_address
		return data[offset : offset + size]
	return b""


def pointer_at(program: Program, address: Address) -> Address | None:
	covering = _object_covering(program, address)
	bound = covering.address + max(covering.size, 1) if covering is not None else None
	return read_pointer(program, address, bound)


def read_pointer(program: Program, address: Address, bound: int | None) -> Address | None:
	"""One pointer read bounded by its enclosing object's exclusive end.

	The bound comes from ``_object_covering`` on the cold path or vsa's
	span bisect, so a pointer that would extend past its object is not
	readable.
	"""
	if bound is not None and address + program.pointer_size > bound:
		return None
	data = memory_at(program, address, program.pointer_size)
	if len(data) != program.pointer_size:
		return None
	return Address(int.from_bytes(data, program.byte_order))
