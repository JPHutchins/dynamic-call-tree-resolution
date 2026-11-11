# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Where each structure or union lies inside a data object, by DWARF's layouts."""

from __future__ import annotations

from enum import StrEnum
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING

from salix import Struct, replace

from dynamic_call_tree_resolution.composition import flatmap
from dynamic_call_tree_resolution.model import ARRAY_SUFFIX

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator, Mapping

	from dynamic_call_tree_resolution.model import (
		DataObject,
		MemberExtent,
		RecordArray,
		StructureLayout,
	)


class RecordInstance(Struct):
	"""A structure or union, by its layout's key, at a byte offset into the object."""

	layout: str
	offset: int


class UnwalkedReason(StrEnum):
	"""Why the layouts can't say which records some bytes hold."""

	UNTYPED = "object without a type"
	NO_LAYOUT = "structure without a layout"
	SIZE = "size unlike its layout's"
	UNSIZED_ARRAY = "array of structures of unknown size"


class Unwalked(Struct):
	"""Bytes from an offset into the object that may hold any record."""

	reason: UnwalkedReason
	offset: int


type Placement = RecordInstance | Unwalked


def object_placements(
	layouts: Mapping[str, StructureLayout], data_object: DataObject
) -> Iterable[Placement]:
	"""Every record the object holds, nested ones included, and any bytes the layouts can't walk.

	>>> from dynamic_call_tree_resolution.model import (
	...     Address,
	...     DataObject,
	...     MemberExtent,
	...     RecordArray,
	...     StructureLayout,
	...     UnnamedMember,
	... )
	>>> leaf = MemberExtent(offset=0, size=4, layout="struct leaf")
	>>> pair = RecordArray(layout="struct leaf", count=2, stride=4)
	>>> layouts = {
	...     "struct leaf": StructureLayout(members=(), size=4),
	...     "union <anonymous>@0x10": StructureLayout(members=(), size=4, extents={"leaf": leaf}),
	...     "struct node": StructureLayout(
	...         members=(),
	...         size=12,
	...         extents={"pair": MemberExtent(offset=4, size=8, layout=None, elements=pair)},
	...         unnamed=(UnnamedMember(offset=0, layout="union <anonymous>@0x10"),),
	...     ),
	... }
	>>> nodes = DataObject(
	...     name="nodes", address=Address(0x100), size=24, type_name="struct node []", signature=None
	... )
	>>> [(placement.layout, placement.offset) for placement in object_placements(layouts, nodes)]
	... # doctest: +NORMALIZE_WHITESPACE
	[('struct node', 0), ('struct leaf', 4), ('struct leaf', 8), ('union <anonymous>@0x10', 0),
	 ('struct leaf', 0), ('struct node', 12), ('struct leaf', 16), ('struct leaf', 20),
	 ('union <anonymous>@0x10', 12), ('struct leaf', 12)]
	>>> from salix import replace
	>>> tuple(object_placements(layouts, replace(nodes, size=20)))
	(Unwalked(reason=<UnwalkedReason.SIZE: "size unlike its layout's">, offset=0),)
	"""
	if data_object.type_name is None:
		return (Unwalked(reason=UnwalkedReason.UNTYPED, offset=0),)
	element = _element_name(data_object.type_name)
	layout = layouts.get(element)
	if layout is None:
		return (
			(Unwalked(reason=UnwalkedReason.NO_LAYOUT, offset=0),) if _names_record(element) else ()
		)
	if not _fits(layout, data_object, whole=element == data_object.type_name):
		return (Unwalked(reason=UnwalkedReason.SIZE, offset=0),)
	return flatmap(
		partial(_placements, layouts, element),
		range(0, data_object.size, layout.size) if layout.size else (0,),
	)


def member_extent(
	layouts: Mapping[str, StructureLayout], key: str, member: str
) -> MemberExtent | None:
	"""A member as C names it, an anonymous member's included, at its offset in the record.

	>>> from dynamic_call_tree_resolution.model import MemberExtent, StructureLayout, UnnamedMember
	>>> layouts = {
	...     "union <anonymous>@0x10": StructureLayout(
	...         members=(), size=4, extents={"next": MemberExtent(offset=0, size=4, layout=None)}
	...     ),
	...     "struct node": StructureLayout(
	...         members=(), size=8, unnamed=(UnnamedMember(offset=4, layout="union <anonymous>@0x10"),)
	...     ),
	... }
	>>> member_extent(layouts, "struct node", "next")
	MemberExtent(offset=4, size=4, layout=None, elements=None)
	>>> member_extent(layouts, "struct node", "prev") is None
	True
	"""
	layout = layouts.get(key)
	if layout is None:
		return None
	own = layout.extents.get(member)
	if own is not None:
		return own
	inside = (
		_shifted(member_extent(layouts, unnamed.layout, member), unnamed.offset)
		for unnamed in layout.unnamed
	)
	return next((extent for extent in inside if extent is not None), None)


def _shifted(extent: MemberExtent | None, offset: int) -> MemberExtent | None:
	return replace(extent, offset=offset + extent.offset) if extent is not None else None


def _element_name(type_name: str) -> str:
	element = type_name.removesuffix(ARRAY_SUFFIX)
	return element if element == type_name else _element_name(element)


def _names_record(type_name: str) -> bool:
	return type_name.startswith(("struct ", "union ")) and not type_name.endswith("*")


def _fits(layout: StructureLayout, data_object: DataObject, *, whole: bool) -> bool:
	return (
		data_object.size == layout.size
		if whole or not layout.size
		else data_object.size % layout.size == 0
	)


def _placements(
	layouts: Mapping[str, StructureLayout], key: str, offset: int
) -> Iterator[Placement]:
	layout = layouts.get(key)
	if layout is None:
		return iter((Unwalked(reason=UnwalkedReason.NO_LAYOUT, offset=offset),))
	return chain(
		(RecordInstance(layout=key, offset=offset),),
		flatmap(partial(_extent_placements, layouts, offset), layout.extents.values()),
		flatmap(
			lambda unnamed: _placements(layouts, unnamed.layout, offset + unnamed.offset),
			layout.unnamed,
		),
	)


def _extent_placements(
	layouts: Mapping[str, StructureLayout], offset: int, extent: MemberExtent
) -> Iterator[Placement]:
	if extent.layout is not None:
		return _sized_placements(layouts, extent.layout, offset + extent.offset, extent.size)
	if extent.elements is None:
		return iter(())
	return _array_placements(layouts, offset + extent.offset, extent.elements)


def _array_placements(
	layouts: Mapping[str, StructureLayout], start: int, elements: RecordArray
) -> Iterator[Placement]:
	if elements.count is None:
		return iter((Unwalked(reason=UnwalkedReason.UNSIZED_ARRAY, offset=start),))
	return flatmap(
		lambda index: _sized_placements(
			layouts, elements.layout, start + index * elements.stride, elements.stride
		),
		range(elements.count),
	)


def _sized_placements(
	layouts: Mapping[str, StructureLayout], key: str, offset: int, size: int | None
) -> Iterator[Placement]:
	layout = layouts.get(key)
	return (
		iter((Unwalked(reason=UnwalkedReason.SIZE, offset=offset),))
		if layout is not None and size is not None and layout.size != size
		else _placements(layouts, key, offset)
	)
