# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""DWARF types: function signatures, type names and structure layouts."""

from __future__ import annotations

from itertools import chain
from math import prod
from typing import TYPE_CHECKING, assert_never

from salix import Struct, replace

from dynamic_call_tree_resolution.composition import merged
from dynamic_call_tree_resolution.loader.dwarf import (
	BLOCK_FORMS,
	CONSTANT_FORMS,
	die_name,
	every_die,
	exprloc,
	int_value,
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
	MemberExtent,
	RecordArray,
	SkippedMember,
	SkipReason,
	StructPointerMember,
	StructureLayout,
	UnnamedMember,
	layout_key,
)

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Iterator

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
			type_name=layout_name(type_die),
			members=tuple(_embedded_members(underlying)),
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
	firsts: dict[str, _Sighting] = {}
	conflicting: set[str] = set()
	for die in every_die(dwarf):
		aggregate = _defined_aggregate(die)
		if aggregate is None:
			continue
		key = layout_name(die)
		members = tuple(map(_held_member, _member_dies(aggregate)))
		layout = StructureLayout(
			members=tuple(_layout_members(members)),
			size=_byte_size(aggregate) or 0,
			extents=_member_extents(members),
			unnamed=tuple(_unnamed_members(members)),
		)
		sighting = _Sighting(aggregate=aggregate, members=members, layout=layout)
		first = firsts.setdefault(key, sighting)
		if (
			first is not sighting
			and key not in conflicting
			and not _same_definition(first, sighting)
		):
			conflicting.add(key)
		existing = layouts.get(key)
		if existing is None or len(layout.members) > len(existing.members):
			layouts[key] = layout
	return {
		key: replace(layout, conflicting=True) if key in conflicting else layout
		for key, layout in layouts.items()
	}


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


class _HeldMember(Struct):
	"""A member's DIE, with what every reading of it needs, resolved once."""

	die: DIE
	name: str | None
	offset: int | None
	type_die: DIE | None
	type_size: int | None
	layout: str | None
	elements: RecordArray | None


def _member_dies(struct_die: DIE) -> Iterator[DIE]:
	return (child for child in struct_die.iter_children() if child.tag == "DW_TAG_member")


def _held_member(member_die: DIE) -> _HeldMember:
	type_die = die_type(member_die)
	type_size = _type_size(type_die)
	return _HeldMember(
		die=member_die,
		name=_member_name(member_die),
		offset=_member_offset(member_die),
		type_die=type_die,
		type_size=type_size,
		layout=_own_layout(type_die),
		elements=_record_array(type_die, type_size),
	)


def _layout_members(members: Iterable[_HeldMember]) -> Iterator[Member]:
	laid_out = map(_layout_member, members)
	return (member for member in laid_out if member is not None)


def _layout_member(held: _HeldMember) -> Member | None:
	return (
		_member(_member_kind(held.type_die), held.name, held.offset)
		if held.offset is not None
		else None
	)


def _embedded_members(struct_die: DIE) -> Iterator[Member]:
	members = map(_embedded_member, _member_dies(struct_die))
	return (member for member in members if member is not None)


def _embedded_member(member_die: DIE) -> Member | None:
	offset = _member_offset(member_die)
	return (
		_member(_member_kind(die_type(member_die)), _member_name(member_die), offset)
		if offset is not None
		else None
	)


def _member_extents(members: Iterable[_HeldMember]) -> dict[str, MemberExtent]:
	return merged(map(_named_extent, members))


def _named_extent(held: _HeldMember) -> dict[str, MemberExtent]:
	return (
		{
			held.name: MemberExtent(
				offset=held.offset,
				size=None if "DW_AT_bit_size" in held.die.attributes else held.type_size,
				layout=held.layout,
				elements=held.elements,
			)
		}
		if held.name is not None and held.offset is not None
		else {}
	)


def _unnamed_members(members: Iterable[_HeldMember]) -> Iterator[UnnamedMember]:
	return (
		UnnamedMember(offset=held.offset, layout=held.layout)
		for held in members
		if held.name is None and held.offset is not None and held.layout is not None
	)


def _record_array(type_die: DIE | None, size: int | None) -> RecordArray | None:
	element_type = _array_element(type_die)
	element = strip_qualifiers(element_type)
	if element is None or element.tag not in ("DW_TAG_structure_type", "DW_TAG_union_type"):
		return None
	stride = _byte_size(element) or 0
	return RecordArray(
		layout=layout_name(element_type),
		count=0 if not stride else size // stride if size is not None else None,
		stride=stride,
	)


def _array_element(type_die: DIE | None) -> DIE | None:
	"""An array's innermost element type, through typedefs of arrays; None for anything else."""
	array = strip_qualifiers(type_die)
	if array is None or array.tag != "DW_TAG_array_type":
		return None
	inner = _array_element(die_type(array))
	return inner if inner is not None else die_type(array)


class _Sighting(Struct):
	"""A unit's definition of a key, as its DIE and as the layout built from it."""

	aggregate: DIE
	members: tuple[_HeldMember, ...]
	layout: StructureLayout


def _same_definition(first: _Sighting, later: _Sighting) -> bool:
	"""Whether two units define a key alike: by definition where an anonymous record's key differs."""
	return (
		_definition(first.aggregate, first.members) == _definition(later.aggregate, later.members)
		if _holds_anonymous(first.layout) or _holds_anonymous(later.layout)
		else first.layout.size == later.layout.size
		and first.layout.extents == later.layout.extents
		and first.layout.unnamed == later.layout.unnamed
	)


def _holds_anonymous(layout: StructureLayout) -> bool:
	keys = chain(
		(extent.layout for extent in layout.extents.values()),
		(
			extent.elements.layout
			for extent in layout.extents.values()
			if extent.elements is not None
		),
		(unnamed.layout for unnamed in layout.unnamed),
	)
	return any(key is not None and f"{ANONYMOUS}@" in key for key in keys)


class _MemberDefinition(Struct):
	"""A member as its record's extents and unnamed members hold it, an anonymous record by definition."""

	name: str | None
	offset: int | None
	size: int | None
	record: str | _Definition | None
	"""The record it holds, itself or as its array's elements: by key, or anonymous by definition."""


class _Definition(Struct):
	"""A record as one unit defines it, so that two units' definitions under one key compare."""

	size: int
	members: tuple[_MemberDefinition, ...]


def _definition(aggregate: DIE, members: Iterable[_HeldMember]) -> _Definition:
	return _Definition(
		size=_byte_size(aggregate) or 0,
		members=tuple(
			_member_definition(held)
			for held in members
			if held.name is not None or held.layout is not None
		),
	)


def _member_definition(held: _HeldMember) -> _MemberDefinition:
	return _MemberDefinition(
		name=held.name,
		offset=held.offset,
		size=None if "DW_AT_bit_size" in held.die.attributes else held.type_size,
		record=_held_record(
			held.type_die if held.elements is None else _array_element(held.type_die)
		),
	)


def _held_record(type_die: DIE | None) -> str | _Definition | None:
	unqualified = strip_qualifiers(type_die)
	if unqualified is None or unqualified.tag not in ("DW_TAG_structure_type", "DW_TAG_union_type"):
		return None
	return (
		_definition(unqualified, map(_held_member, _member_dies(unqualified)))
		if _anonymous_aggregate(unqualified)
		else layout_name(type_die)
	)


def _type_size(type_die: DIE | None) -> int | None:
	unqualified = strip_qualifiers(type_die)
	return (
		_array_size(unqualified)
		if unqualified is not None
		and unqualified.tag == "DW_TAG_array_type"
		and "DW_AT_byte_size" not in unqualified.attributes
		else _byte_size(unqualified)
	)


def _array_size(array_die: DIE) -> int | None:
	element = _type_size(die_type(array_die))
	counts = _known_counts(array_die)
	return element * prod(counts) if element and counts is not None else None


def _own_layout(type_die: DIE | None) -> str | None:
	unqualified = strip_qualifiers(type_die)
	return (
		layout_name(type_die)
		if unqualified is not None
		and unqualified.tag in ("DW_TAG_structure_type", "DW_TAG_union_type")
		else None
	)


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
