# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Static resolution of function-pointer slots to candidate target functions."""

from __future__ import annotations

from bisect import bisect_right
from itertools import pairwise
from typing import TYPE_CHECKING, assert_never

from salix import Struct

from dynamic_call_tree_resolution.model import (
	ARRAY_SUFFIX,
	FUNCTION_POINTER,
	Address,
	ArrayMember,
	EmbeddedStructMember,
	FunctionPointerMember,
	FunctionSignature,
	InstructionSet,
	Machine,
	NotEnumerated,
	Program,
	Provenance,
	Residue,
	Section,
	SkippedMember,
	SkipReason,
	SlotAssignment,
	StructPointerMember,
	UnresolvedSlot,
	array_element_type,
	array_elements,
)

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator, Mapping

	from dynamic_call_tree_resolution.model import DataObject, Member, StructureLayout


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


def in_writable_memory(program: Program, address: Address) -> bool:
	covering = _section_covering(program, address)
	return covering is None or covering[1].writable


def _from_relocations(program: Program) -> Iterable[_Resolution]:
	for relocation in program.relocations:
		if relocation.target not in program.functions:
			continue
		yield _Resolution(
			slot=relocation.slot,
			path=_slot_path(program, relocation.slot),
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
				target_object = _object_covering(
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
				UnresolvedSlot(
					slot=slot,
					path=entry.path,
					signature=entry.signature,
					residue=_residue(program, slot),
				)
				for slot, entry in universe.items()
				if slot not in resolved_by_slot
			),
			key=_slot_address,
		)
	)


def _residue(program: Program, slot: Address) -> Residue:
	value = pointer_at(program, slot)
	if not in_writable_memory(program, slot):
		return Residue.ROM_NULL if value == 0 else Residue.ROM_NON_FUNCTION
	return (
		Residue.RAM_UNINITIALIZED
		if value is None
		else Residue.RAM_NULL
		if value == 0
		else Residue.RAM_INITIALIZED
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
			case ArrayMember(offset=offset):
				_collect_member_slots(
					data_object, base_offset + offset, array_elements(member), member_path, universe
				)
			case StructPointerMember() | SkippedMember():
				pass
			case _ as unreachable:
				assert_never(unreachable)


def not_enumerated(program: Program) -> tuple[NotEnumerated, ...]:
	return tuple(
		skipped
		for data_object in sorted(program.objects.values(), key=_object_address)
		for skipped in _object_skips(program, data_object)
	)


def _object_skips(program: Program, data_object: DataObject) -> Iterator[NotEnumerated]:
	layout = _layout_of(program, data_object)
	if layout is not None:
		yield from _skipped_members(layout.members, (data_object.name,))
		return
	element_layout = (
		program.layouts.get(array_element_type(data_object.type_name))
		if data_object.type_name is not None and data_object.type_name.endswith(ARRAY_SUFFIX)
		else None
	)
	if element_layout is not None and element_layout.size > 0:
		for index in range(data_object.size // element_layout.size):
			yield from _skipped_members(element_layout.members, (data_object.name, f"[{index}]"))
	elif _untyped_and_unchecked(program, data_object):
		yield NotEnumerated(path=(data_object.name,), reason=SkipReason.UNTYPED_OBJECT)


def _untyped_and_unchecked(program: Program, data_object: DataObject) -> bool:
	words = tuple(
		pointer_at(program, Address(data_object.address + base_offset))
		for base_offset in range(0, data_object.size, program.pointer_size)
	)
	functions = sum(
		1 for word in words if word is not None and word != 0 and word in program.functions
	)
	return (
		data_object.type_name is None
		and data_object.size > program.pointer_size
		and functions < len(words)
		and (functions > 0 or in_writable_memory(program, data_object.address))
	)


def _skipped_members(
	members: tuple[Member, ...], path: tuple[str | None, ...]
) -> Iterator[NotEnumerated]:
	for member in members:
		member_path = (*path, member.name)
		match member:
			case SkippedMember(reason=reason):
				yield NotEnumerated(path=member_path, reason=reason)
			case EmbeddedStructMember(members=inner_members):
				yield from _skipped_members(inner_members, member_path)
			case ArrayMember():
				yield from _skipped_members(array_elements(member), member_path)
			case FunctionPointerMember() | StructPointerMember():
				pass
			case _ as unreachable:
				assert_never(unreachable)


def _object_address(data_object: DataObject) -> Address:
	return data_object.address


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
	return {slot.slot: slot.signature for slot in unresolved if slot.signature is not None}


def memory_at(program: Program, address: Address, size: int) -> bytes:
	covering = _section_covering(program, address)
	if covering is None:
		return b""
	start, section = covering
	return section.data[address - start : address - start + size]


def instruction_runs(
	program: Program, start: Address, size: int
) -> tuple[tuple[Address, bytes], ...]:
	end = start + size
	edges = (
		start,
		*(
			edge
			for low, high in sorted(program.data_in_code)
			if low < end and start < high
			for edge in (max(low, start), min(high, end))
		),
		end,
	)
	return tuple(
		(Address(low), memory_at(program, Address(low), high - low))
		for code_start, code_end in zip(edges[::2], edges[1::2], strict=True)
		for low, high in pairwise(
			(
				code_start,
				*sorted(
					edge
					for span in program.arm_code
					for edge in span
					if code_start < edge < code_end
				),
				code_end,
			)
		)
		if high > low
	)


def instruction_set_at(program: Program, address: Address) -> InstructionSet:
	match program.machine:
		case Machine.EM_X86_64:
			return InstructionSet.X86_64
		case Machine.EM_386:
			return InstructionSet.X86_32
		case Machine.EM_ARM:
			index = bisect_right(program.arm_code, address, key=_span_start) - 1
			return (
				InstructionSet.A32
				if index >= 0 and address < program.arm_code[index][1]
				else InstructionSet.T32
			)
		case _ as unreachable:
			assert_never(unreachable)


def _span_start(span: tuple[Address, Address]) -> Address:
	return span[0]


def _section_covering(program: Program, address: Address) -> tuple[Address, Section] | None:
	return next(
		(
			(start, section)
			for start, section in program.sections.items()
			if start <= address < start + len(section.data)
		),
		None,
	)


def pointer_at(program: Program, address: Address) -> Address | None:
	covering = _object_covering(program, address)
	bound = covering.address + max(covering.size, 1) if covering is not None else None
	return read_pointer(program, address, bound)


def read_pointer(program: Program, address: Address, bound: int | None) -> Address | None:
	if bound is not None and address + program.pointer_size > bound:
		return None
	data = memory_at(program, address, program.pointer_size)
	if len(data) != program.pointer_size:
		return None
	return Address(int.from_bytes(data, program.byte_order))
