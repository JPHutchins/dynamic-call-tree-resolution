# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Naming the bytes an assumption flag covers: a data object, or a member of one."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

import pytest
from salix import replace

from dynamic_call_tree_resolution.cli.expansion import named_span
from dynamic_call_tree_resolution.model import (
	Address,
	DataObject,
	Machine,
	MemberExtent,
	StructureLayout,
)
from tests.programs import build_program, span

if TYPE_CHECKING:
	from dynamic_call_tree_resolution.model import Program

_OBJECTS: Final = (
	("cfg", 0x2000, 16, "struct cfg"),
	("CSWTCH.88", 0x2100, 4, None),
	("small", 0x2200, 8, "struct small"),
	("twin", 0x2300, 4, None),
	("twin", 0x2400, 4, None),
)


def _program() -> Program:
	return replace(
		build_program(
			Machine.EM_ARM,
			objects=tuple((name, address, bytes(size)) for name, address, size, _ in _OBJECTS),
			pointer_size=4,
		),
		objects={
			Address(address): DataObject(
				name=name,
				address=Address(address),
				size=size,
				type_name=type_name,
				signature=None,
			)
			for name, address, size, type_name in _OBJECTS
		},
		layouts={
			"struct cfg": StructureLayout(
				members=(),
				size=16,
				extents={
					"dev": MemberExtent(offset=0, size=4, layout=None),
					"inner": MemberExtent(offset=4, size=8, layout="struct inner"),
					"bits": MemberExtent(offset=12, size=None, layout=None),
				},
			),
			"struct inner": StructureLayout(
				members=(), size=8, extents={"fn": MemberExtent(offset=4, size=4, layout=None)}
			),
			"struct small": StructureLayout(
				members=(), size=4, extents={"x": MemberExtent(offset=0, size=4, layout=None)}
			),
		},
	)


@pytest.mark.parametrize(
	("name", "named"),
	[
		pytest.param("cfg", (0x2000, 0x2010), id="a whole object"),
		pytest.param("cfg.dev", (0x2000, 0x2004), id="a member"),
		pytest.param("cfg.inner.fn", (0x2008, 0x200C), id="a member of an embedded structure"),
		pytest.param("CSWTCH.88", (0x2100, 0x2104), id="a symbol with a dot in its name"),
	],
)
def test_a_name_covers_its_object_or_member(name: str, named: tuple[int, int]) -> None:
	assert named_span(_program(), "--flag", name) == span(*named)


@pytest.mark.parametrize(
	("name", "message"),
	[
		pytest.param("nothing", "no data object is named nothing", id="no such object"),
		pytest.param("twin", "2 data objects are named twin", id="a name two objects share"),
		pytest.param(
			"cfg.nope", "DWARF places no member nope in cfg's struct cfg", id="no such member"
		),
		pytest.param(
			"cfg.dev.x",
			"DWARF gives cfg.dev no structure, so no member x",
			id="a member of a scalar",
		),
		pytest.param(
			"cfg.bits",
			"DWARF gives cfg.bits no size; name what holds it",
			id="a member with no size",
		),
		pytest.param(
			"small.x",
			"small is 8 bytes, but DWARF's struct small is 4; name what holds it",
			id="a layout of another size",
		),
	],
)
def test_a_name_dwarf_cannot_place_fails(name: str, message: str) -> None:
	with pytest.raises(ValueError, match=re.escape(f"--flag: {message}")):
		named_span(_program(), "--flag", name)
