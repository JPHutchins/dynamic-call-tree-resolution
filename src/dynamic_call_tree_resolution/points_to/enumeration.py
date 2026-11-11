# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The data objects the slot walk skips, and why."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, assert_never

from dynamic_call_tree_resolution.composition import flatmap
from dynamic_call_tree_resolution.model import (
	ARRAY_SUFFIX,
	Address,
	ArrayMember,
	EmbeddedStructMember,
	FunctionPointerMember,
	NotEnumerated,
	Program,
	SkippedMember,
	SkipReason,
	StructPointerMember,
	array_element_type,
	array_elements,
)
from dynamic_call_tree_resolution.points_to.image import in_writable_memory, pointer_at
from dynamic_call_tree_resolution.points_to.objects import layout_of

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator

	from dynamic_call_tree_resolution.model import DataObject, Member


def not_enumerated(program: Program) -> tuple[NotEnumerated, ...]:
	return tuple(
		flatmap(
			partial(_object_skips, program),
			sorted(program.objects.values(), key=_object_address),
		)
	)


def _object_skips(program: Program, data_object: DataObject) -> Iterable[NotEnumerated]:
	layout = layout_of(program, data_object)
	if layout is not None:
		return _skipped_members(layout.members, (data_object.name,))
	element_layout = (
		program.layouts.get(array_element_type(data_object.type_name))
		if data_object.type_name is not None and data_object.type_name.endswith(ARRAY_SUFFIX)
		else None
	)
	if element_layout is not None and element_layout.size > 0:
		return flatmap(
			lambda index: _skipped_members(
				element_layout.members, (data_object.name, f"[{index}]")
			),
			range(data_object.size // element_layout.size),
		)
	if _untyped_and_unchecked(program, data_object):
		return (NotEnumerated(path=(data_object.name,), reason=SkipReason.UNTYPED_OBJECT),)
	return ()


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
	return flatmap(partial(_skips_in_member, path), members)


def _skips_in_member(path: tuple[str | None, ...], member: Member) -> Iterable[NotEnumerated]:
	member_path = (*path, member.name)
	match member:
		case SkippedMember(reason=reason):
			return (NotEnumerated(path=member_path, reason=reason),)
		case EmbeddedStructMember(members=inner_members):
			return _skipped_members(inner_members, member_path)
		case ArrayMember():
			return _skipped_members(array_elements(member), member_path)
		case FunctionPointerMember() | StructPointerMember():
			return ()
		case _ as unreachable:
			assert_never(unreachable)


def _object_address(data_object: DataObject) -> Address:
	return data_object.address
