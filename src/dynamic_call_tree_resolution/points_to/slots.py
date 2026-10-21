# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The slots no assignment resolves, and the slots of each struct field."""

from __future__ import annotations

from itertools import groupby
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
	NullSlot,
	Program,
	Residue,
	SkippedMember,
	SlotAssignment,
	StructPointerMember,
	UnresolvedSlot,
	array_element_type,
	array_elements,
)
from dynamic_call_tree_resolution.points_to.image import in_writable_memory, pointer_at
from dynamic_call_tree_resolution.points_to.objects import layout_of, slot_path

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping

	from dynamic_call_tree_resolution.model import DataObject, Member


class _SlotUniverseEntry(Struct):
	path: tuple[str | None, ...]
	signature: FunctionSignature | None
	record: str | None = None
	"""The layout whose member the slot is."""


def unresolved_slots(
	program: Program, resolved: tuple[SlotAssignment, ...]
) -> tuple[UnresolvedSlot, ...]:
	resolved_by_slot = {assignment.slot for assignment in resolved}
	return tuple(
		sorted(
			(
				UnresolvedSlot(
					slot=slot,
					path=entry.path,
					signature=entry.signature,
					residue=_residue(program, slot),
				)
				for slot, entry in dict(reversed(tuple(_slot_universe(program)))).items()
				if slot not in resolved_by_slot and not _holds_null_in_rom(program, slot)
			),
			key=_slot_address,
		)
	)


def null_slots(program: Program, resolved: tuple[SlotAssignment, ...]) -> tuple[NullSlot, ...]:
	resolved_by_slot = {assignment.slot for assignment in resolved}
	return tuple(
		sorted(
			(
				NullSlot(slot=slot, path=entry.path, signature=entry.signature)
				for slot, entry in dict(reversed(tuple(_slot_universe(program)))).items()
				if slot not in resolved_by_slot and _holds_null_in_rom(program, slot)
			),
			key=_slot_address,
		)
	)


def field_slots(program: Program) -> Mapping[tuple[str, str], frozenset[Address]]:
	"""Every function-pointer slot that is a named member of a layout, by layout and member."""
	return {
		field: frozenset(slot for _, slot in group)
		for field, group in groupby(
			sorted(
				((entry.record, member), slot)
				for slot, entry in _slot_universe(program)
				if entry.record is not None
				for member in (entry.path[-1],)
				if member is not None
			),
			key=_field_of,
		)
	}


def signatures_by_slot(
	unresolved: tuple[UnresolvedSlot, ...],
) -> Mapping[Address, FunctionSignature]:
	return {slot.slot: slot.signature for slot in unresolved if slot.signature is not None}


def _slot_address(slot: UnresolvedSlot | NullSlot) -> Address:
	return slot.slot


def _holds_null_in_rom(program: Program, slot: Address) -> bool:
	return not in_writable_memory(program, slot) and pointer_at(program, slot) == 0


def _field_of(entry: tuple[tuple[str, str], Address]) -> tuple[str, str]:
	return entry[0]


def _slot_universe(program: Program) -> Iterable[tuple[Address, _SlotUniverseEntry]]:
	for data_object in program.objects.values():
		layout = layout_of(program, data_object)
		if layout is None:
			if data_object.type_name == FUNCTION_POINTER:
				yield (
					data_object.address,
					_SlotUniverseEntry(path=(data_object.name,), signature=data_object.signature),
				)
			elif data_object.type_name is not None and data_object.type_name.endswith(ARRAY_SUFFIX):
				yield from _array_element_slots(program, data_object, (data_object.name,))
			continue
		yield from _member_slots(
			data_object, 0, layout.members, (data_object.name,), data_object.type_name
		)
	for relocation in program.relocations:
		if relocation.target not in program.functions:
			continue
		yield (
			relocation.slot,
			_SlotUniverseEntry(path=slot_path(program, relocation.slot), signature=None),
		)


def _residue(program: Program, slot: Address) -> Residue:
	value = pointer_at(program, slot)
	if not in_writable_memory(program, slot):
		return Residue.ROM_NON_FUNCTION
	return (
		Residue.RAM_UNINITIALIZED
		if value is None
		else Residue.RAM_NULL
		if value == 0
		else Residue.RAM_INITIALIZED
	)


def _array_element_slots(
	program: Program, data_object: DataObject, path: tuple[str | None, ...]
) -> Iterable[tuple[Address, _SlotUniverseEntry]]:
	element_name = array_element_type(data_object.type_name) if data_object.type_name else ""
	element_layout = program.layouts.get(element_name)
	if element_layout is None:
		if element_name == FUNCTION_POINTER:
			for index, base_offset in enumerate(range(0, data_object.size, program.pointer_size)):
				yield (
					Address(data_object.address + base_offset),
					_SlotUniverseEntry(path=(*path, f"[{index}]"), signature=data_object.signature),
				)
		return
	for index, base_offset in enumerate(range(0, data_object.size, element_layout.size or 1)):
		yield from _member_slots(
			data_object, base_offset, element_layout.members, (*path, f"[{index}]"), element_name
		)


def _member_slots(
	data_object: DataObject,
	base_offset: int,
	members: tuple[Member, ...],
	path: tuple[str | None, ...],
	record: str | None,
) -> Iterable[tuple[Address, _SlotUniverseEntry]]:
	for member in members:
		member_path = (*path, member.name)
		match member:
			case FunctionPointerMember(offset=offset, signature=signature):
				yield (
					Address(data_object.address + base_offset + offset),
					_SlotUniverseEntry(path=member_path, signature=signature, record=record),
				)
			case EmbeddedStructMember(offset=offset, type_name=type_name, members=inner_members):
				yield from _member_slots(
					data_object, base_offset + offset, inner_members, member_path, type_name
				)
			case ArrayMember(offset=offset):
				yield from _member_slots(
					data_object, base_offset + offset, array_elements(member), member_path, None
				)
			case StructPointerMember() | SkippedMember():
				pass
			case _ as unreachable:
				assert_never(unreachable)
