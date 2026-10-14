# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The image's data objects, from its symbol table and DWARF."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from salix import replace

from dynamic_call_tree_resolution.loader.dwarf import (
	BLOCK_FORMS,
	DW_OP_ADDR,
	exprloc,
	int_value,
	iter_dies,
	string_value,
)
from dynamic_call_tree_resolution.loader.layouts import (
	FunctionPointer,
	Skipped,
	StructPointer,
	die_type,
	layout_name,
	pointee_kind,
	strip_qualifiers,
)
from dynamic_call_tree_resolution.model import (
	Address,
	DataObject,
	FunctionSignature,
)

if TYPE_CHECKING:
	from collections.abc import Mapping

	from elftools.dwarf.die import DIE
	from elftools.dwarf.dwarfinfo import DWARFInfo
	from elftools.elf.sections import SymbolTableSection

	from dynamic_call_tree_resolution.model import ByteOrder


def objects_from_symtab(symtab: SymbolTableSection | None) -> dict[Address, DataObject]:
	if symtab is None:
		return {}  # pragma: no cover
	objects: dict[Address, DataObject] = {}
	for symbol in symtab.iter_symbols():
		if symbol["st_info"]["type"] != "STT_OBJECT":
			continue
		if symbol["st_shndx"] == "SHN_UNDEF":
			continue  # pragma: no cover
		if symbol["st_shndx"] == "SHN_ABS":
			continue
		address = Address(symbol["st_value"])
		if objects.get(address) is None or symbol["st_size"] > objects[address].size:
			objects[address] = DataObject(
				name=symbol.name,
				address=address,
				size=symbol["st_size"],
				type_name=None,
				signature=None,
			)
	return objects


def objects_from_dwarf(
	dwarf: DWARFInfo | None,
	pointer_size: int,
	byte_order: ByteOrder,
) -> dict[Address, DataObject]:
	if dwarf is None:
		return {}
	return {
		address: DataObject(
			name=string_value(name_attribute),
			address=address,
			size=_byte_size_of(die),
			type_name=layout_name(die_type(die)),
			signature=_object_signature(die),
		)
		for compilation_unit in dwarf.iter_CUs()
		for die in iter_dies(compilation_unit.get_top_DIE())
		if die.tag == "DW_TAG_variable"
		if "DW_AT_declaration" not in die.attributes
		if (name_attribute := die.attributes.get("DW_AT_name")) is not None
		if (address := _location_address(die, pointer_size, byte_order)) is not None
	}


def _object_signature(die: DIE) -> FunctionSignature | None:
	type_die = die_type(die)
	if type_die is not None and type_die.tag == "DW_TAG_array_type":
		type_die = die_type(type_die)
	match pointee_kind(type_die):
		case FunctionPointer(signature):
			return signature
		case StructPointer() | Skipped() | None:
			return None
		case _ as unreachable:
			assert_never(unreachable)


def _location_address(die: DIE, pointer_size: int, byte_order: ByteOrder) -> Address | None:
	location = die.attributes.get("DW_AT_location")
	if location is None:
		return None
	if location.form not in BLOCK_FORMS:
		return None  # pragma: no cover
	operations = exprloc(location)
	if not operations or operations[0] != DW_OP_ADDR:
		return None  # pragma: no cover
	return Address(int.from_bytes(bytes(operations[1 : 1 + pointer_size]), byte_order))


def _byte_size_of(die: DIE) -> int:
	type_die = strip_qualifiers(die_type(die))
	if type_die is None:
		return 0  # pragma: no cover
	byte_size = type_die.attributes.get("DW_AT_byte_size")
	return int_value(byte_size) if byte_size is not None else 0  # pragma: no branch


def types_of_declarations(dwarf: DWARFInfo | None) -> dict[str, str]:
	if dwarf is None:
		return {}
	return {
		string_value(name_attribute): layout_name(die_type(die))
		for compilation_unit in dwarf.iter_CUs()
		for die in iter_dies(compilation_unit.get_top_DIE())
		if die.tag == "DW_TAG_variable"
		if "DW_AT_declaration" in die.attributes
		if (name_attribute := die.attributes.get("DW_AT_name")) is not None
	}


def merge_objects(
	symtab_objects: dict[Address, DataObject],
	dwarf_objects: dict[Address, DataObject],
	declaration_types: Mapping[str, str],
) -> dict[Address, DataObject]:
	merged = dict(dwarf_objects)
	for address, data_object in symtab_objects.items():
		if address not in merged:
			merged[address] = data_object
		if merged[address].type_name is None and data_object.name in declaration_types:
			merged[address] = replace(
				merged[address], type_name=declaration_types[data_object.name]
			)
		if data_object.size > merged[address].size:
			merged[address] = replace(merged[address], size=data_object.size)
	return merged
