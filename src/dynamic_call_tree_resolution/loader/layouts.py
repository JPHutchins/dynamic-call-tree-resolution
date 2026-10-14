# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""DWARF types: function signatures, type names and structure layouts."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, assert_never

from salix import Struct

from dynamic_call_tree_resolution.loader.dwarf import (
	BLOCK_FORMS,
	CONSTANT_FORMS,
	die_name,
	exprloc,
	int_value,
	iter_dies,
	plus_uconst,
	string_value,
)
from dynamic_call_tree_resolution.model import (
	ANONYMOUS,
	ARRAY_SUFFIX,
	FUNCTION_POINTER,
	ArrayMember,
	EmbeddedStructMember,
	FunctionPointerMember,
	FunctionSignature,
	SkippedMember,
	SkipReason,
	StructPointerMember,
	StructureLayout,
	layout_key,
)

if TYPE_CHECKING:
	from collections.abc import Callable, Iterator

	from elftools.dwarf.die import DIE, AttributeValue
	from elftools.dwarf.dwarfinfo import DWARFInfo

	from dynamic_call_tree_resolution.model import Member


def function_signature(die: DIE) -> FunctionSignature:
	return FunctionSignature(
		return_type=_type_name(die_type(die)),
		parameters=tuple(
			_type_name(parameter_die)
			if parameter_die is not None
			else "<unknown>"  # pragma: no branch
			for parameter_die in (
				die_type(child)
				for child in die.iter_children()
				if child.tag == "DW_TAG_formal_parameter"
			)
		),
	)


def _type_name(die: DIE | None) -> str:
	return _named_type(die, _shared_anonymous)


def layout_name(die: DIE | None) -> str:
	return _named_type(die, _distinct_anonymous)


def _shared_anonymous(_: DIE) -> str:
	return ANONYMOUS


def _distinct_anonymous(die: DIE) -> str:
	return f"{ANONYMOUS}@{die.offset:#x}"


def _named_type(die: DIE | None, anonymous: Callable[[DIE], str]) -> str:
	if die is None:
		return "void"
	match die.tag:
		case "DW_TAG_base_type" | "DW_TAG_enumeration_type":
			return die_name(die)
		case (
			"DW_TAG_const_type"
			| "DW_TAG_volatile_type"
			| "DW_TAG_restrict_type"
			| "DW_TAG_atomic_type"
		):
			return _named_type(die_type(die), anonymous)
		case "DW_TAG_pointer_type":
			pointee = strip_qualifiers(die_type(die))
			if pointee is None:
				return "void *"
			return (
				FUNCTION_POINTER
				if pointee.tag == "DW_TAG_subroutine_type"
				else f"{_named_type(die_type(die), anonymous)} *"
			)
		case "DW_TAG_typedef":
			return (
				die_name(die)
				if _anonymous_aggregate(die_type(die))
				else _named_type(die_type(die), anonymous)
			)
		case "DW_TAG_structure_type" | "DW_TAG_union_type" | "DW_TAG_class_type":
			return layout_key(
				"union" if die.tag == "DW_TAG_union_type" else "struct",
				die_name(die) if "DW_AT_name" in die.attributes else anonymous(die),
			)
		case "DW_TAG_array_type":
			return f"{_named_type(die_type(die), anonymous)}{ARRAY_SUFFIX}"
		case "DW_TAG_subroutine_type":  # pragma: no cover
			return FUNCTION_POINTER
		case _:
			return str(die.tag)  # pragma: no cover


def _anonymous_aggregate(die: DIE | None) -> bool:
	return (
		die is not None
		and die.tag in ("DW_TAG_structure_type", "DW_TAG_union_type")
		and "DW_AT_name" not in die.attributes
	)


def die_type(die: DIE) -> DIE | None:
	if "DW_AT_type" not in die.attributes:
		return None
	return die.get_DIE_from_attribute("DW_AT_type")


def strip_qualifiers(die: DIE | None) -> DIE | None:
	if die is None or die.tag not in (
		"DW_TAG_const_type",
		"DW_TAG_volatile_type",
		"DW_TAG_restrict_type",
		"DW_TAG_atomic_type",
		"DW_TAG_typedef",
	):
		return die
	return strip_qualifiers(die_type(die))


class FunctionPointer(Struct):
	"""A pointer to a function, by its signature."""

	signature: FunctionSignature


class StructPointer(Struct):
	"""A pointer to a structure, by its layout's name."""

	pointee: str | None


class _EmbeddedStruct(Struct):
	type_name: str
	members: tuple[Member, ...]


class _Array(Struct):
	count: int
	stride: int
	element: Member


class Skipped(Struct):
	"""A pointee the slot enumeration skips, and why."""

	reason: SkipReason


type _PointeeKind = FunctionPointer | StructPointer | Skipped | None


type _MemberKind = FunctionPointer | StructPointer | _EmbeddedStruct | _Array | Skipped | None


def pointee_kind(type_die: DIE | None) -> _PointeeKind:
	underlying = strip_qualifiers(type_die)
	if underlying is None or underlying.tag != "DW_TAG_pointer_type":
		return None  # pragma: no cover
	pointee = strip_qualifiers(die_type(underlying))
	if pointee is None:
		return StructPointer(pointee=None)
	match pointee.tag:
		case "DW_TAG_subroutine_type":
			return FunctionPointer(signature=function_signature(pointee))
		case "DW_TAG_structure_type" | "DW_TAG_union_type":
			return StructPointer(pointee=layout_name(die_type(underlying)))
		case "DW_TAG_pointer_type" if pointee_kind(pointee) is not None:
			return Skipped(reason=SkipReason.POINTER_TO_POINTER)
		case "DW_TAG_array_type" if _array_kind(pointee) is not None:
			return Skipped(reason=SkipReason.POINTER_TO_ARRAY)
		case _:
			return None


def _member_kind(type_die: DIE | None) -> _MemberKind:
	underlying = strip_qualifiers(type_die)
	if underlying is None:
		return None  # pragma: no cover
	if underlying.tag == "DW_TAG_pointer_type":
		return pointee_kind(underlying)
	if underlying.tag in ("DW_TAG_structure_type", "DW_TAG_union_type"):
		return _EmbeddedStruct(
			type_name=layout_name(type_die), members=tuple(_layout_members(underlying))
		)
	if underlying.tag == "DW_TAG_array_type":
		return _array_kind(underlying)
	return None


def _array_kind(array_die: DIE) -> _Array | Skipped | None:
	element = _member(_member_kind(die_type(array_die)), None, 0)
	if element is None:
		return None
	stride = _byte_size(strip_qualifiers(die_type(array_die)))
	counts = _known_counts(array_die)
	if not stride or counts is None:
		return Skipped(reason=SkipReason.UNSIZED_ARRAY)
	return _nested_array(counts, stride, element)


def _known_counts(array_die: DIE) -> tuple[int, ...] | None:
	counts = tuple(
		_subrange_count(child)
		for child in array_die.iter_children()
		if child.tag == "DW_TAG_subrange_type"
	)
	return tuple(count for count in counts if count) if counts and all(counts) else None


def _nested_array(counts: tuple[int, ...], stride: int, element: Member) -> _Array:
	return _Array(
		count=counts[0],
		stride=stride * prod(counts[1:]),
		element=element
		if len(counts) == 1
		else _array_member(_nested_array(counts[1:], stride, element), None, 0),
	)


def _subrange_count(subrange: DIE) -> int | None:
	count = _constant(subrange.attributes.get("DW_AT_count"))
	upper = _constant(subrange.attributes.get("DW_AT_upper_bound"))
	lower = _constant(subrange.attributes.get("DW_AT_lower_bound")) or 0
	return count if count is not None else None if upper is None else upper - lower + 1


def _constant(attribute: AttributeValue | None) -> int | None:
	return (
		int_value(attribute) if attribute is not None and attribute.form in CONSTANT_FORMS else None
	)


def _byte_size(die: DIE | None) -> int | None:
	return _constant(die.attributes.get("DW_AT_byte_size")) if die is not None else None


def _member(kind: _MemberKind, name: str | None, offset: int) -> Member | None:
	match kind:
		case FunctionPointer(signature):
			return FunctionPointerMember(
				kind="function_pointer", name=name, offset=offset, signature=signature
			)
		case StructPointer(pointee):
			return StructPointerMember(
				kind="struct_pointer", name=name, offset=offset, pointee=pointee
			)
		case _EmbeddedStruct(type_name, members):
			return EmbeddedStructMember(
				kind="embedded_struct",
				name=name,
				offset=offset,
				type_name=type_name,
				members=members,
			)
		case _Array() as array:
			return _array_member(array, name, offset)
		case Skipped(reason):
			return SkippedMember(kind="skipped", name=name, offset=offset, reason=reason)
		case None:
			return None
		case _ as unreachable:
			assert_never(unreachable)


def _array_member(array: _Array, name: str | None, offset: int) -> ArrayMember:
	return ArrayMember(
		kind="array",
		name=name,
		offset=offset,
		count=array.count,
		stride=array.stride,
		element=array.element,
	)


def structure_layouts(dwarf: DWARFInfo | None) -> dict[str, StructureLayout]:
	if dwarf is None:
		return {}
	layouts: dict[str, StructureLayout] = {}
	for compilation_unit in dwarf.iter_CUs():
		for die in iter_dies(compilation_unit.get_top_DIE()):
			aggregate = _defined_aggregate(die)
			if aggregate is None:
				continue
			key = layout_name(die)
			layout = StructureLayout(
				members=tuple(_layout_members(aggregate)),
				size=_byte_size(aggregate) or 0,
				offsets=_member_offsets(aggregate),
			)
			existing = layouts.get(key)
			if existing is None or len(layout.members) > len(existing.members):
				layouts[key] = layout
	return layouts


def _defined_aggregate(die: DIE) -> DIE | None:
	aggregate = (
		die
		if die.tag in ("DW_TAG_structure_type", "DW_TAG_union_type")
		else die_type(die)
		if die.tag == "DW_TAG_typedef" and _anonymous_aggregate(die_type(die))
		else None
	)
	return (
		aggregate
		if aggregate is not None and "DW_AT_declaration" not in aggregate.attributes
		else None
	)


def _layout_members(struct_die: DIE) -> Iterator[Member]:
	for child in struct_die.iter_children():
		if child.tag != "DW_TAG_member":
			continue  # pragma: no cover
		offset = _member_offset(child)
		if offset is None:
			continue  # pragma: no cover
		member = _member(_member_kind(die_type(child)), _member_name(child), offset)
		if member is not None:
			yield member


def _member_offsets(struct_die: DIE) -> dict[str, int]:
	return {
		name: offset
		for child in struct_die.iter_children()
		if child.tag == "DW_TAG_member"
		and (name := _member_name(child)) is not None
		and (offset := _member_offset(child)) is not None
	}


def _member_name(member_die: DIE) -> str | None:
	name_attribute = member_die.attributes.get("DW_AT_name")
	return None if name_attribute is None else string_value(name_attribute)


def _member_offset(member_die: DIE) -> int | None:
	location = member_die.attributes.get("DW_AT_data_member_location")
	if location is None:
		return 0
	if location.form in CONSTANT_FORMS:
		return int_value(location)
	return plus_uconst(exprloc(location)) if location.form in BLOCK_FORMS else None
