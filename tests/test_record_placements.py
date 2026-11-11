# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Every structure or union a data object holds, by DWARF's layouts: tests/fixtures/record_placements.c."""

from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import TYPE_CHECKING, assert_never

import pytest

from dynamic_call_tree_resolution.loader import load
from dynamic_call_tree_resolution.model import (
	Address,
	DataObject,
	MemberExtent,
	RecordArray,
	StructureLayout,
)
from dynamic_call_tree_resolution.record_placements import (
	RecordInstance,
	Unwalked,
	UnwalkedReason,
	member_extent,
	object_placements,
)

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path

	from dynamic_call_tree_resolution.record_placements import Placement


def test_a_layout_lists_its_unnamed_records_and_its_arrays_of_records(
	fixture_elfs: dict[str, Path],
) -> None:
	layouts = load(fixture_elfs["placements"]).layouts
	node = layouts["struct node"]
	assert [(unnamed.offset, _kind(unnamed.layout)) for unnamed in node.unnamed] == [
		(0, "union <anonymous>"),
		(8, "union <anonymous>"),
	]
	assert [
		(unnamed.offset, _kind(unnamed.layout))
		for unnamed in layouts[node.unnamed[1].layout].unnamed
	] == [(0, "struct <anonymous>")]
	assert {name: extent.elements for name, extent in node.extents.items()} == {
		"flags": None,
		"leaf": None,
		"pair": RecordArray(layout="struct leaf", count=2, stride=16),
		"grid": RecordArray(layout="struct leaf", count=6, stride=16),
		"nothing": None,
	}
	assert {name: extent.elements for name, extent in layouts["struct rows"].extents.items()} == {
		"rows": RecordArray(layout="struct leaf", count=4, stride=16),
		"count": None,
		"tail": RecordArray(layout="struct leaf", count=None, stride=16),
	}


def test_a_member_of_an_anonymous_member_lies_at_its_offset_in_the_record(
	fixture_elfs: dict[str, Path],
) -> None:
	layouts = load(fixture_elfs["placements"]).layouts
	assert {
		member: member_extent(layouts, "struct node", member)
		for member in ("head", "prio", "locked", "both", "leaf", "missing")
	} == {
		"head": MemberExtent(offset=0, size=8, layout=None),
		"prio": MemberExtent(offset=8, size=1, layout=None),
		"locked": MemberExtent(offset=9, size=1, layout=None),
		"both": MemberExtent(offset=8, size=2, layout=None),
		"leaf": MemberExtent(offset=16, size=16, layout="struct leaf"),
		"missing": None,
	}
	assert member_extent(layouts, "struct missing", "head") is None


def test_an_object_places_every_record_it_holds(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["placements"])
	objects = {data_object.name: data_object for data_object in program.objects.values()}
	placed = {
		name: tuple(object_placements(program.layouts, objects[name]))
		for name in ("nodes", "rows", "arms", "counter", "leaves")
	}
	assert {name: Counter(map(_kind_of, placements)) for name, placements in placed.items()} == {
		"nodes": Counter(
			{
				"struct leaf": 18,
				"union <anonymous>": 4,
				"struct node": 2,
				"struct empty": 2,
				"struct <anonymous>": 2,
			}
		),
		"rows": Counter({"struct leaf": 4, "struct rows": 1, UnwalkedReason.UNSIZED_ARRAY: 1}),
		"arms": Counter(
			{
				"struct leaf": 10,
				"union <anonymous>": 2,
				"union arms": 1,
				"struct node": 1,
				"struct empty": 1,
				"struct <anonymous>": 1,
			}
		),
		"counter": Counter(),
		"leaves": Counter(),
	}
	assert sorted(
		placement.offset
		for placement in placed["nodes"]
		if isinstance(placement, RecordInstance) and placement.layout == "struct leaf"
	) == list(chain(range(16, 160, 16), range(176, 320, 16)))
	assert Unwalked(reason=UnwalkedReason.UNSIZED_ARRAY, offset=72) in placed["rows"]


def test_a_key_two_units_define_differently_is_unwalked_and_one_they_share_is_not(
	fixture_elfs: dict[str, Path],
) -> None:
	program = load(fixture_elfs["placements"])
	objects = {data_object.name: data_object for data_object in program.objects.values()}
	assert {
		key: program.layouts[key].conflicting
		for key in ("struct ctx", "struct node", "struct leaf", "struct empty")
	} == {"struct ctx": True, "struct node": False, "struct leaf": False, "struct empty": False}
	assert {
		name: tuple(object_placements(program.layouts, objects[name]))
		for name in ("first_ctx", "second_ctx")
	} == {
		"first_ctx": (Unwalked(reason=UnwalkedReason.CONFLICTING, offset=0),),
		"second_ctx": (Unwalked(reason=UnwalkedReason.CONFLICTING, offset=0),),
	}
	assert Counter(map(_kind_of, object_placements(program.layouts, objects["other_nodes"]))) == (
		Counter(
			{
				"struct leaf": 9,
				"union <anonymous>": 2,
				"struct node": 1,
				"struct empty": 1,
				"struct <anonymous>": 1,
			}
		)
	)


_LEAF: StructureLayout = StructureLayout(members=(), size=16)


def _holder(inner: str, size: int) -> StructureLayout:
	return StructureLayout(
		members=(), size=16, extents={"inner": MemberExtent(offset=8, size=size, layout=inner)}
	)


@pytest.mark.parametrize(
	("type_name", "size", "layouts", "unwalked"),
	[
		(None, 16, {}, Unwalked(reason=UnwalkedReason.UNTYPED, offset=0)),
		("struct missing", 16, {}, Unwalked(reason=UnwalkedReason.NO_LAYOUT, offset=0)),
		("struct leaf", 8, {"struct leaf": _LEAF}, Unwalked(reason=UnwalkedReason.SIZE, offset=0)),
		(
			"struct leaf []",
			24,
			{"struct leaf": _LEAF},
			Unwalked(reason=UnwalkedReason.SIZE, offset=0),
		),
		(
			"struct leaf []",
			0,
			{"struct leaf": _LEAF},
			Unwalked(reason=UnwalkedReason.UNSIZED_ARRAY, offset=0),
		),
		(
			"struct holder",
			16,
			{"struct holder": _holder("struct missing", 8)},
			Unwalked(reason=UnwalkedReason.NO_LAYOUT, offset=8),
		),
		(
			"struct holder",
			16,
			{"struct holder": _holder("struct leaf", 8), "struct leaf": _LEAF},
			Unwalked(reason=UnwalkedReason.SIZE, offset=8),
		),
	],
)
def test_bytes_the_layouts_cannot_walk_are_unwalked(
	type_name: str | None, size: int, layouts: Mapping[str, StructureLayout], unwalked: Unwalked
) -> None:
	data_object = DataObject(
		name="held", address=Address(0x100), size=size, type_name=type_name, signature=None
	)
	assert [
		placement
		for placement in object_placements(layouts, data_object)
		if isinstance(placement, Unwalked)
	] == [unwalked]


def test_an_object_of_scalars_or_pointers_holds_no_record() -> None:
	assert [
		tuple(
			object_placements(
				{},
				DataObject(
					name="held", address=Address(0x100), size=8, type_name=name, signature=None
				),
			)
		)
		for name in ("int", "struct leaf *", "struct leaf * []", "function pointer")
	] == [(), (), (), ()]


def _kind_of(placement: Placement) -> str:
	match placement:
		case RecordInstance(layout=layout):
			return _kind(layout)
		case Unwalked(reason=reason):
			return reason
		case _ as unreachable:
			assert_never(unreachable)


def _kind(layout: str) -> str:
	return layout.split("@", 1)[0]
