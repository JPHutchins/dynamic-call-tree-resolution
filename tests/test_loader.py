# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Tests for :mod:`dynamic_call_tree_resolution.loader`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dynamic_call_tree_resolution import (
	EmbeddedStructMember,
	FunctionPointerMember,
	FunctionSignature,
	StructPointerMember,
	load,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_load_functions_from_dwarf_and_symtab(fixture_elfs: dict[str, Path]) -> None:
	names = {function.name for function in load(fixture_elfs["pie"]).functions.values()}
	assert {"driver_a_open", "driver_b_close", "plain_target", "main"} <= names
	assert "puts" not in names


def test_load_without_dwarf(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["nodebug"])
	names = {function.name for function in program.functions.values()}
	assert "driver_a_open" in names
	assert "plain_target" in names
	assert program.layouts == {}


def test_load_layouts(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["pie"])
	assert set(program.layouts) == {
		"struct ops",
		"struct device",
		"struct device_ops",
		"struct handler_holder",
		"struct node",
		"struct container",
		"struct anon_wrapper",
		"struct embedded_holder",
		"union un",
		"struct bitpacked",
	}
	assert program.layouts["struct ops"].members == (
		FunctionPointerMember(
			kind="function_pointer",
			name="open",
			offset=0,
			signature=FunctionSignature(return_type="int", parameters=("void *", "int")),
		),
		FunctionPointerMember(
			kind="function_pointer",
			name="close",
			offset=8,
			signature=FunctionSignature(return_type="int", parameters=("void *",)),
		),
	)
	assert program.layouts["struct device"].members == (
		StructPointerMember(kind="struct_pointer", name="api", offset=0, pointee="struct ops"),
		StructPointerMember(kind="struct_pointer", name="context", offset=8, pointee=None),
		EmbeddedStructMember(
			kind="embedded_struct",
			name="ops",
			offset=16,
			members=(
				FunctionPointerMember(
					kind="function_pointer",
					name="init",
					offset=0,
					signature=FunctionSignature(return_type="int", parameters=()),
				),
			),
		),
	)
	assert program.layouts["struct embedded_holder"].members == (
		EmbeddedStructMember(
			kind="embedded_struct",
			name="inner",
			offset=0,
			members=(
				FunctionPointerMember(
					kind="function_pointer",
					name="fn",
					offset=8,
					signature=FunctionSignature(return_type="int", parameters=()),
				),
			),
		),
	)
	assert program.layouts["struct bitpacked"].members == ()


def test_load_declaration_types_attach_to_symtab_objects(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["multi"])
	dev_x = next(
		data_object for data_object in program.objects.values() if data_object.name == "dev_x"
	)
	assert dev_x.type_name == "struct device"


def test_load_objects(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["pie"])
	by_name = {data_object.name: data_object for data_object in program.objects.values()}
	assert by_name["dev_a"].type_name == "struct device"
	assert len(by_name["dev_a"].bytes) == 24
	assert by_name["plain_cb"].type_name == "function pointer"
	assert by_name["bss_holder"].bytes == b""
	assert program.pointer_size == 8
	assert program.byte_order == "little"
