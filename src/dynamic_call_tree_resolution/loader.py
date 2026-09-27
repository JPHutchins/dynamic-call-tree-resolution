# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Loading ELF and DWARF data into the immutable :class:`Program` model."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from elftools.elf.descriptions import describe_reloc_type
from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import RelocationSection
from elftools.elf.sections import SymbolTableSection
from salix import replace

from dynamic_call_tree_resolution.model import (
	Address,
	DataObject,
	EmbeddedStructMember,
	Function,
	FunctionPointerMember,
	FunctionSignature,
	Program,
	Relocation,
	StructPointerMember,
	StructureLayout,
)

if TYPE_CHECKING:
	from collections.abc import Iterator, Mapping
	from pathlib import Path
	from typing import BinaryIO, Literal

	from elftools.dwarf.die import DIE
	from elftools.dwarf.dwarfinfo import DWARFInfo

	from dynamic_call_tree_resolution.model import ByteOrder, Member


class _SectionBytes(NamedTuple):
	address: int
	size: int
	data: bytes
	flags: int


_DW_OP_ADDR = 0x03
_SHF_ALLOC = 0x2


def load(path: Path) -> Program:
	"""Parse an ELF file at ``path`` into a :class:`Program`.

	Malformed input propagates the underlying parser's errors unchanged.
	"""
	with path.open("rb") as stream:
		return _load(stream)


def _load(stream: BinaryIO) -> Program:
	elf = ELFFile(stream)
	dwarf = elf.get_dwarf_info() if elf.has_dwarf_info() else None
	symtab = _symbol_table(elf)
	relocations = _relocations(elf)
	sections = _relocated_sections(_sections(elf), relocations, elf.elfclass // 8, _byte_order(elf))
	return Program(
		byte_order=_byte_order(elf),
		pointer_size=elf.elfclass // 8,
		machine=elf.header["e_machine"],
		functions={**_functions_from_symtab(symtab), **_functions_from_dwarf(dwarf)},
		objects=_merge_objects(
			_objects_from_symtab(symtab),
			_objects_from_dwarf(dwarf, elf.elfclass // 8, _byte_order(elf)),
			_declaration_types(dwarf),
		),
		layouts=_layouts(dwarf),
		relocations=relocations,
		sections={
			Address(section.address): section.data
			for section in sections.values()
			if section.flags & _SHF_ALLOC
		},
	)


def _byte_order(elf: ELFFile) -> ByteOrder:
	return "little" if elf.little_endian else "big"  # pragma: no branch


def _symbol_table(elf: ELFFile) -> SymbolTableSection | None:
	section = elf.get_section_by_name(".symtab")
	return section if isinstance(section, SymbolTableSection) else None  # pragma: no branch


def _sections(elf: ELFFile) -> dict[str, _SectionBytes]:
	return {
		section.name: _SectionBytes(
			address=section.header.sh_addr,
			size=section.header.sh_size,
			data=section.data(),
			flags=section.header.sh_flags,
		)
		for section in elf.iter_sections()
		if section.header.sh_size > 0
		and section.header.sh_type != "SHT_NOBITS"
		and not section.name.startswith((".debug", ".symtab", ".strtab"))
	}


def _relocated_sections(
	sections: dict[str, _SectionBytes],
	relocations: tuple[Relocation, ...],
	pointer_size: int,
	byte_order: ByteOrder,
) -> dict[str, _SectionBytes]:
	patches = {
		relocation.slot: relocation.target.to_bytes(pointer_size, byte_order)
		for relocation in relocations
		if relocation.target != 0
	}
	return {
		name: _SectionBytes(
			address=section.address,
			size=section.size,
			data=_patched(section, patches, pointer_size),
			flags=section.flags,
		)
		for name, section in sections.items()
	}


def _patched(section: _SectionBytes, patches: Mapping[Address, bytes], pointer_size: int) -> bytes:
	data = bytearray(section.data)
	for slot, target in patches.items():
		if section.address <= slot < section.address + section.size:
			offset = slot - section.address
			data[offset : offset + pointer_size] = target
	return bytes(data)


def _functions_from_symtab(symtab: SymbolTableSection | None) -> dict[Address, Function]:
	if symtab is None:
		return {}  # pragma: no cover
	functions: dict[Address, Function] = {}
	for symbol in symtab.iter_symbols():
		if symbol["st_info"]["type"] != "STT_FUNC":
			continue
		if symbol["st_shndx"] == "SHN_UNDEF":
			continue
		address = Address(symbol["st_value"])
		functions[address] = Function(
			name=symbol.name,
			address=address,
			size=symbol["st_size"],
			signature=None,
		)
	return functions


def _functions_from_dwarf(dwarf: DWARFInfo | None) -> dict[Address, Function]:
	if dwarf is None:
		return {}  # pragma: no cover
	functions: dict[Address, Function] = {}
	for compilation_unit in dwarf.iter_CUs():
		for die in _iter_dies(compilation_unit.get_top_DIE()):
			if die.tag != "DW_TAG_subprogram":
				continue
			low_pc = die.attributes.get("DW_AT_low_pc")
			if low_pc is None:
				continue
			address = Address(int(low_pc.value))
			if address & ~1 == 0:
				continue
			functions[address] = Function(
				name=_die_name(die),
				address=address,
				size=_subprogram_size(die),
				signature=_signature(die),
			)
	return functions


def _subprogram_size(die: DIE) -> int:
	low_pc = int(die.attributes["DW_AT_low_pc"].value)
	high_pc = die.attributes.get("DW_AT_high_pc")
	if high_pc is None:
		return 0  # pragma: no cover
	if high_pc.form == "DW_FORM_addr":  # pragma: no branch
		return int(high_pc.value) - low_pc  # pragma: no cover
	return int(high_pc.value)


def _signature(die: DIE) -> FunctionSignature:
	return FunctionSignature(
		return_type=_type_name(_type_die(die)),
		parameters=tuple(
			_type_name(parameter_die)
			if parameter_die is not None
			else "<unknown>"  # pragma: no branch
			for parameter_die in (
				_type_die(child)
				for child in die.iter_children()
				if child.tag == "DW_TAG_formal_parameter"
			)
		),
	)


def _type_name(die: DIE | None) -> str:
	if die is None:
		return "void"
	match die.tag:
		case "DW_TAG_base_type" | "DW_TAG_enumeration_type":
			return _die_name(die)
		case "DW_TAG_const_type" | "DW_TAG_volatile_type":
			return _type_name(_type_die(die))
		case "DW_TAG_pointer_type":
			pointee = _strip_qualifiers(_type_die(die))
			if pointee is None:
				return "void *"
			return (
				"function pointer"
				if pointee.tag == "DW_TAG_subroutine_type"
				else f"{_type_name(pointee)} *"
			)
		case "DW_TAG_typedef":
			return _type_name(_type_die(die))
		case "DW_TAG_structure_type" | "DW_TAG_union_type" | "DW_TAG_class_type":
			return f"{'union' if die.tag == 'DW_TAG_union_type' else 'struct'} {_die_name(die)}"
		case "DW_TAG_subroutine_type":  # pragma: no cover
			return "function pointer"
		case _:
			return str(die.tag)


def _type_die(die: DIE) -> DIE | None:
	if "DW_AT_type" not in die.attributes:
		return None
	return die.get_DIE_from_attribute("DW_AT_type")


def _strip_qualifiers(die: DIE | None) -> DIE | None:
	if die is None or die.tag not in (
		"DW_TAG_const_type",
		"DW_TAG_volatile_type",
		"DW_TAG_typedef",
	):
		return die
	return _strip_qualifiers(_type_die(die))


def _pointee_kind(
	type_die: DIE | None,
) -> (
	tuple[Literal["function_pointer"], FunctionSignature]
	| tuple[Literal["struct"], str | None]
	| None
):
	underlying = _strip_qualifiers(type_die)
	if underlying is None or underlying.tag != "DW_TAG_pointer_type":
		return None  # pragma: no cover
	pointee = _strip_qualifiers(_type_die(underlying))
	if pointee is None:
		return ("struct", None)
	match pointee.tag:
		case "DW_TAG_subroutine_type":
			return ("function_pointer", _signature(pointee))
		case "DW_TAG_structure_type" | "DW_TAG_union_type" if "DW_AT_name" in pointee.attributes:
			return ("struct", _type_name(pointee))
		case _:
			return None


def _member_kind(
	type_die: DIE | None,
) -> (
	tuple[Literal["function_pointer"], FunctionSignature]
	| tuple[Literal["struct"], str | None]
	| tuple[Literal["embedded"], tuple[Member, ...]]
	| None
):
	underlying = _strip_qualifiers(type_die)
	if underlying is None:
		return None  # pragma: no cover
	if underlying.tag == "DW_TAG_pointer_type":
		return _pointee_kind(underlying)
	if underlying.tag in ("DW_TAG_structure_type", "DW_TAG_union_type"):
		return ("embedded", tuple(_layout_members(underlying)))
	return None


def _layouts(dwarf: DWARFInfo | None) -> dict[str, StructureLayout]:
	if dwarf is None:
		return {}  # pragma: no cover
	layouts: dict[str, StructureLayout] = {}
	for compilation_unit in dwarf.iter_CUs():
		for die in _iter_dies(compilation_unit.get_top_DIE()):
			if die.tag not in ("DW_TAG_structure_type", "DW_TAG_union_type"):
				continue
			name_attribute = die.attributes.get("DW_AT_name")
			if name_attribute is None:
				continue
			name = _attr_string(name_attribute.value)
			keyword = "union" if die.tag == "DW_TAG_union_type" else "struct"
			layouts[f"{keyword} {name}"] = StructureLayout(members=tuple(_layout_members(die)))
	return layouts


def _layout_members(struct_die: DIE) -> Iterator[Member]:
	for child in struct_die.iter_children():
		if child.tag != "DW_TAG_member":
			continue  # pragma: no cover
		offset = _member_offset(child)
		if offset is None:
			continue
		match _member_kind(_type_die(child)):
			case ("function_pointer", signature):
				yield FunctionPointerMember(
					kind="function_pointer",
					name=_member_name(child),
					offset=offset,
					signature=signature,
				)
			case ("struct", pointee):
				yield StructPointerMember(
					kind="struct_pointer",
					name=_member_name(child),
					offset=offset,
					pointee=pointee,
				)
			case ("embedded", members):
				yield EmbeddedStructMember(
					kind="embedded_struct",
					name=_member_name(child),
					offset=offset,
					members=members,
				)
			case None:  # pragma: no branch
				pass


def _member_name(member_die: DIE) -> str | None:
	name_attribute = member_die.attributes.get("DW_AT_name")
	return None if name_attribute is None else _attr_string(name_attribute.value)


def _member_offset(member_die: DIE) -> int | None:
	location = member_die.attributes.get("DW_AT_data_member_location")
	if location is None or location.form not in (
		"DW_FORM_data1",
		"DW_FORM_data2",
		"DW_FORM_data4",
		"DW_FORM_data8",
		"DW_FORM_udata",
	):
		return None  # pragma: no branch
	return int(location.value)


def _objects_from_symtab(symtab: SymbolTableSection | None) -> dict[Address, DataObject]:
	if symtab is None:
		return {}  # pragma: no cover
	objects: dict[Address, DataObject] = {}
	for symbol in symtab.iter_symbols():
		if symbol["st_info"]["type"] != "STT_OBJECT":
			continue
		if symbol["st_shndx"] == "SHN_UNDEF":
			continue  # pragma: no cover
		address = Address(symbol["st_value"])
		objects[address] = DataObject(
			name=symbol.name,
			address=address,
			size=symbol["st_size"],
			type_name=None,
			signature=None,
		)
	return objects


def _objects_from_dwarf(
	dwarf: DWARFInfo | None,
	pointer_size: int,
	byte_order: ByteOrder,
) -> dict[Address, DataObject]:
	if dwarf is None:
		return {}  # pragma: no cover
	objects: dict[Address, DataObject] = {}
	for compilation_unit in dwarf.iter_CUs():
		for die in _iter_dies(compilation_unit.get_top_DIE()):
			if die.tag != "DW_TAG_variable":
				continue
			if "DW_AT_declaration" in die.attributes:
				continue
			name_attribute = die.attributes.get("DW_AT_name")
			address = _location_address(die, pointer_size, byte_order)
			if name_attribute is None or address is None:
				continue  # pragma: no branch
			size = _byte_size_of(die)
			objects[address] = DataObject(
				name=_attr_string(name_attribute.value),
				address=address,
				size=size,
				type_name=_type_name(_type_die(die)),
				signature=_object_signature(die),
			)
	return objects


def _object_signature(die: DIE) -> FunctionSignature | None:
	match _pointee_kind(_type_die(die)):
		case ("function_pointer", signature):
			return signature
		case _:
			return None


def _location_address(die: DIE, pointer_size: int, byte_order: ByteOrder) -> Address | None:
	location = die.attributes.get("DW_AT_location")
	if location is None:
		return None
	if location.form not in (
		"DW_FORM_exprloc",
		"DW_FORM_block1",
		"DW_FORM_block2",
		"DW_FORM_block4",
	):
		return None  # pragma: no cover
	operations = location.value
	if not operations or operations[0] != _DW_OP_ADDR:
		return None  # pragma: no cover
	return Address(int.from_bytes(bytes(operations[1 : 1 + pointer_size]), byte_order))


def _byte_size_of(die: DIE) -> int:
	type_die = _strip_qualifiers(_type_die(die))
	if type_die is None:
		return 0  # pragma: no cover
	byte_size = type_die.attributes.get("DW_AT_byte_size")
	return int(byte_size.value) if byte_size is not None else 0  # pragma: no branch


def _declaration_types(dwarf: DWARFInfo | None) -> dict[str, str]:
	if dwarf is None:
		return {}  # pragma: no cover
	declarations: dict[str, str] = {}
	for compilation_unit in dwarf.iter_CUs():
		for die in _iter_dies(compilation_unit.get_top_DIE()):
			if die.tag != "DW_TAG_variable":
				continue
			if "DW_AT_declaration" not in die.attributes:
				continue
			name_attribute = die.attributes.get("DW_AT_name")
			if name_attribute is None:
				continue  # pragma: no cover
			declarations[_attr_string(name_attribute.value)] = _type_name(_type_die(die))
	return declarations


def _merge_objects(
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
	return merged


def _relocations(elf: ELFFile) -> tuple[Relocation, ...]:
	relocations: list[Relocation] = []
	for section in elf.iter_sections():
		if not isinstance(section, RelocationSection):
			continue
		symbol_table = elf.get_section(section["sh_link"])
		if not isinstance(symbol_table, SymbolTableSection):
			continue  # pragma: no cover
		for relocation in section.iter_relocations():
			target = (
				relocation["r_addend"]
				if relocation["r_info_sym"] == 0
				else symbol_table.get_symbol(relocation["r_info_sym"])["st_value"]
			)
			relocations.append(
				Relocation(
					slot=Address(relocation["r_offset"]),
					target=Address(target),
					addend=relocation["r_addend"] if section.is_RELA() else 0,  # pragma: no branch
					type_name=describe_reloc_type(relocation["r_info_type"], elf),
				)
			)
	return tuple(relocations)


def _iter_dies(root: DIE) -> Iterator[DIE]:
	yield root
	for child in root.iter_children():
		yield from _iter_dies(child)


def _die_name(die: DIE) -> str:
	for attribute_name in ("DW_AT_linkage_name", "DW_AT_name"):
		attribute = die.attributes.get(attribute_name)
		if attribute is not None:
			return _attr_string(attribute.value)
	return "<anonymous>"


def _attr_string(value: str | bytes) -> str:
	return value if isinstance(value, str) else value.decode()  # pragma: no branch
