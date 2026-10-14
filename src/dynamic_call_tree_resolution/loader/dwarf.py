# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Reading attributes and locations from DWARF debugging entries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from dynamic_call_tree_resolution.model import (
	ANONYMOUS,
)

if TYPE_CHECKING:
	from collections.abc import Iterator

	from elftools.dwarf.die import DIE, AttributeValue


DW_OP_ADDR: Final = 0x03


_DW_OP_PLUS_UCONST: Final = 0x23


CONSTANT_FORMS: Final = frozenset(
	{
		"DW_FORM_data1",
		"DW_FORM_data2",
		"DW_FORM_data4",
		"DW_FORM_data8",
		"DW_FORM_udata",
		"DW_FORM_sdata",
		"DW_FORM_implicit_const",
	}
)


BLOCK_FORMS: Final = frozenset(
	{"DW_FORM_exprloc", "DW_FORM_block1", "DW_FORM_block2", "DW_FORM_block4"}
)


def iter_dies(root: DIE) -> Iterator[DIE]:
	yield root
	for child in root.iter_children():
		yield from iter_dies(child)


def int_value(attribute: AttributeValue) -> int:
	return int(cast("int | str | bytes", attribute.value))


def string_value(attribute: AttributeValue) -> str:
	return attr_string(cast("str | bytes", attribute.value))


def exprloc(attribute: AttributeValue) -> list[int]:
	return cast("list[int]", attribute.value)


def die_name(die: DIE) -> str:
	for attribute_name in ("DW_AT_linkage_name", "DW_AT_name"):
		attribute = die.attributes.get(attribute_name)
		if attribute is not None:
			return string_value(attribute)
	origin = die_origin(die)
	return ANONYMOUS if origin is None else die_name(origin)


def die_origin(die: DIE) -> DIE | None:
	return next(
		(
			die.get_DIE_from_attribute(attribute_name)
			for attribute_name in ("DW_AT_abstract_origin", "DW_AT_specification")
			if attribute_name in die.attributes
		),
		None,
	)


def declaration(die: DIE) -> DIE:
	origin = die_origin(die)
	return die if origin is None else declaration(origin)


def attr_string(value: str | bytes) -> str:
	return value if isinstance(value, str) else value.decode()  # pragma: no branch


def _uleb128(data: list[int]) -> int:
	return sum(
		(byte & 0x7F) << (7 * index)
		for index, byte in enumerate(
			data[
				: next((index for index, byte in enumerate(data) if not byte & 0x80), len(data)) + 1
			]
		)
	)


def plus_uconst(expression: list[int]) -> int | None:
	"""The offset a DWARF 2 member location adds to its struct's address.

	>>> plus_uconst([0x23, 0x08]), plus_uconst([0x23, 0x90, 0x01]), plus_uconst([0x10, 0x08])
	(8, 144, None)
	"""
	return _uleb128(expression[1:]) if expression[:1] == [_DW_OP_PLUS_UCONST] else None
