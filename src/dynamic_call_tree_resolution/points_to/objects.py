# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The data objects and DWARF layouts that hold an image's slots."""

from __future__ import annotations

import heapq
from bisect import bisect_right
from itertools import groupby
from typing import TYPE_CHECKING

from salix import Struct

from dynamic_call_tree_resolution.composition import flatmap

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator

	from dynamic_call_tree_resolution.model import (
		Address,
		DataObject,
		Member,
		Program,
		StructureLayout,
	)


def slot_path(program: Program, address: Address) -> tuple[str | None, ...]:
	return owned_slot_path(program, object_covering(program, address), address)


def owned_slot_path(
	program: Program, owner: DataObject | None, address: Address
) -> tuple[str | None, ...]:
	if owner is None:
		return ()  # pragma: no cover
	layout = layout_of(program, owner)
	if layout is None:
		return (owner.name,)
	member = _member_at(layout, address - owner.address)
	return (owner.name, member.name) if member is not None else (owner.name,)  # pragma: no branch


def object_covering(program: Program, address: Address) -> DataObject | None:
	return next(
		(
			data_object
			for data_object in program.objects.values()
			if data_object.address <= address < data_object.address + max(data_object.size, 1)
		),
		None,
	)


class ObjectIndex(Struct):
	"""The object ``object_covering`` finds for each address."""

	boundaries: tuple[int, ...]
	"""Every object's first address and the address past its last."""
	covering: tuple[DataObject | None, ...]
	"""The object covering the addresses from each boundary up to the next."""


def object_index(program: Program) -> ObjectIndex:
	return object_index_of(program.objects.values())


def object_index_of(objects: Iterable[DataObject]) -> ObjectIndex:
	"""Each address owned by the first of the objects that covers it, as ``object_covering`` finds it.

	>>> from dynamic_call_tree_resolution.model import Address, DataObject
	>>> def held(name, address, size):
	...     return DataObject(
	...         name=name, address=Address(address), size=size, type_name=None, signature=None
	...     )
	>>> index = object_index_of(
	...     (
	...         held("inner", 0x108, 4),
	...         held("outer", 0x100, 16),
	...         held("mark", 0x120, 0),
	...         held("alias", 0x100, 0),
	...     )
	... )
	>>> [
	...     getattr(covering_object(index, Address(address)), "name", None)
	...     for address in (0xFF, 0x100, 0x108, 0x10C, 0x110, 0x120, 0x121)
	... ]
	[None, 'outer', 'inner', 'outer', None, 'mark', None]
	"""
	ranked = tuple(
		_Ranked(rank=rank, data_object=data_object) for rank, data_object in enumerate(objects)
	)
	boundaries = tuple(sorted(frozenset(flatmap(_edges, ranked))))
	return ObjectIndex(boundaries=boundaries, covering=tuple(_sweep(ranked, boundaries)))


def covering_object(index: ObjectIndex, address: Address | None) -> DataObject | None:
	if address is None:
		return None
	position = bisect_right(index.boundaries, address) - 1
	return index.covering[position] if position >= 0 else None


def object_end(data_object: DataObject) -> int:
	return data_object.address + max(data_object.size, 1)


class _Ranked(Struct, order=True):
	"""An object, ranked by its place in ``program.objects``, which decides between overlapping ones."""

	rank: int
	data_object: DataObject


def _edges(ranked: _Ranked) -> tuple[int, ...]:
	return (_start(ranked), _end(ranked))


def _start(ranked: _Ranked) -> int:
	return ranked.data_object.address


def _end(ranked: _Ranked) -> int:
	return object_end(ranked.data_object)


def _sweep(ranked: Iterable[_Ranked], boundaries: Iterable[int]) -> Iterator[DataObject | None]:
	starting = {
		start: tuple(group) for start, group in groupby(sorted(ranked, key=_start), key=_start)
	}
	active: list[_Ranked] = []
	for boundary in boundaries:
		for entry in starting.get(boundary, ()):
			heapq.heappush(active, entry)
		while active and _end(active[0]) <= boundary:
			heapq.heappop(active)
		yield active[0].data_object if active else None


def layout_of(program: Program, data_object: DataObject) -> StructureLayout | None:
	if data_object.type_name is None:
		return None
	return program.layouts.get(data_object.type_name)


def _member_at(layout: StructureLayout, offset: int) -> Member | None:
	return next(
		(member for member in layout.members if member.offset == offset), None
	)  # pragma: no branch
