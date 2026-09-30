# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Loading ELF and DWARF data into the immutable :class:`Program` model."""

from __future__ import annotations

import re
from itertools import groupby
from typing import TYPE_CHECKING, Final, assert_never, cast

from elftools.elf.descriptions import describe_reloc_type
from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import Relocation as ElfRelocation
from elftools.elf.relocation import RelocationSection
from elftools.elf.sections import SymbolTableSection
from salix import Struct, replace

from dynamic_call_tree_resolution.model import (
	ANONYMOUS,
	ARRAY_SUFFIX,
	FUNCTION_POINTER,
	Address,
	DataObject,
	EmbeddedStructMember,
	Function,
	FunctionPointerMember,
	FunctionSignature,
	Machine,
	Program,
	Relocation,
	Section,
	StructPointerMember,
	StructureLayout,
	layout_key,
	thumb_twin,
)

if TYPE_CHECKING:
	from collections.abc import Iterator, Mapping
	from pathlib import Path
	from typing import BinaryIO

	from elftools.dwarf.die import DIE, AttributeValue
	from elftools.dwarf.dwarfinfo import DWARFInfo
	from elftools.elf.sections import Symbol

	from dynamic_call_tree_resolution.model import ByteOrder, Member


class _SectionBytes(Struct):
	address: int
	size: int
	data: bytes
	flags: int


_DW_OP_ADDR: Final = 0x03
_SHF_WRITE: Final = 0x1
_SHF_ALLOC: Final = 0x2
_SHF_EXECINSTR: Final = 0x4
_MAPPING_SYMBOL: Final = re.compile(r"\$[adt](\..*)?")


def load(path: Path) -> Program:
	with path.open("rb") as stream:
		return _load(stream)


def defined_function_names(path: Path) -> frozenset[str]:
	with path.open("rb") as stream:
		symtab = _symbol_table(ELFFile(stream))
		return frozenset(
			symbol.name
			for symbol in (symtab.iter_symbols() if symtab is not None else ())
			if symbol["st_info"]["type"] == "STT_FUNC"
			if symbol["st_shndx"] != "SHN_UNDEF"
		)


def _load(stream: BinaryIO) -> Program:
	elf = ELFFile(stream)
	if elf.header["e_type"] == "ET_REL":
		raise ValueError("relocatable (ET_REL) images are not supported; link the image first")
	dwarf = elf.get_dwarf_info() if elf.has_dwarf_info() else None
	symtab = _symbol_table(elf)
	pointer_size = elf.elfclass // 8
	byte_order = _byte_order(elf)
	raw_sections = _sections(elf)
	relocations = _relocations(elf, raw_sections, pointer_size, byte_order)
	sections = _relocated_sections(raw_sections, relocations, pointer_size, byte_order)
	relro_spans = tuple(
		(segment["p_vaddr"], segment["p_vaddr"] + segment["p_memsz"])
		for segment in elf.iter_segments(type="PT_GNU_RELRO")
	)
	machine = Machine(elf.header["e_machine"])
	markers = _mapping_markers(symtab)
	return Program(
		byte_order=byte_order,
		pointer_size=pointer_size,
		machine=machine,
		functions=_merge_functions(
			_functions_from_symtab(symtab), _functions_from_dwarf(dwarf), machine
		),
		objects=_merge_objects(
			_objects_from_symtab(symtab),
			_objects_from_dwarf(dwarf, pointer_size, byte_order),
			_declaration_types(dwarf),
		),
		layouts=_layouts(dwarf),
		relocations=relocations,
		sections={
			Address(section.address): Section(
				data=section.data,
				writable=bool(section.flags & _SHF_WRITE) and not _relro(relro_spans, section),
			)
			for section in sections.values()
			if section.flags & _SHF_ALLOC
		},
		data_in_code=_mapped_spans(elf, markers, "d"),
		arm_code=tuple(sorted(_mapped_spans(elf, markers, "a"))),
	)


def _mapped_spans(
	elf: ELFFile, markers: tuple[tuple[int, int, str], ...], kind: str
) -> tuple[tuple[Address, Address], ...]:
	return tuple(
		span
		for section_index, group in groupby(markers, key=_marker_section)
		if (section := elf.get_section(section_index)) is not None
		and section.header.sh_flags & _SHF_EXECINSTR
		for span in _kind_spans(tuple(group), section.header.sh_addr + section.header.sh_size, kind)
	)


def _mapping_markers(symtab: SymbolTableSection | None) -> tuple[tuple[int, int, str], ...]:
	return tuple(
		sorted(
			marker
			for symbol in (symtab.iter_symbols() if symtab is not None else ())
			if (marker := _mapping_marker(symbol)) is not None
		)
	)


def _mapping_marker(symbol: Symbol) -> tuple[int, int, str] | None:
	section_index = symbol["st_shndx"]
	if not _MAPPING_SYMBOL.fullmatch(symbol.name) or not isinstance(section_index, int):
		return None
	return section_index, symbol["st_value"], symbol.name[1]


def _marker_section(marker: tuple[int, int, str]) -> int:
	return marker[0]


def _kind_spans(
	marks: tuple[tuple[int, int, str], ...], end: int, kind: str
) -> Iterator[tuple[Address, Address]]:
	return (
		(Address(address), Address(following))
		for (_, address, mark_kind), following in zip(
			marks, (*(mark[1] for mark in marks[1:]), end), strict=True
		)
		if mark_kind == kind and following > address
	)


def _relro(spans: tuple[tuple[int, int], ...], section: _SectionBytes) -> bool:
	return any(
		start <= section.address and section.address + section.size <= end for start, end in spans
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
	return {
		address: Function(
			name=symbol.name,
			address=address,
			size=symbol["st_size"],
			signature=None,
		)
		for symbol in sorted(symtab.iter_symbols(), key=_symbol_size)
		if symbol["st_info"]["type"] == "STT_FUNC"
		if symbol["st_shndx"] != "SHN_UNDEF"
		if (address := Address(symbol["st_value"])) & ~1
	}


def _merge_functions(
	from_symtab: Mapping[Address, Function],
	from_dwarf: Mapping[Address, Function],
	machine: Machine,
) -> dict[Address, Function]:
	return {
		**from_symtab,
		**{
			merged.address: merged
			for merged in (
				_at_symbol(function, from_symtab, machine) for function in from_dwarf.values()
			)
		},
	}


def _at_symbol(
	function: Function, from_symtab: Mapping[Address, Function], machine: Machine
) -> Function:
	address = (
		thumb_twin(function.address)
		if machine is Machine.EM_ARM and thumb_twin(function.address) in from_symtab
		else function.address
	)
	symbol = from_symtab.get(address)
	return Function(
		name=symbol.name if symbol is not None else function.name,
		address=address,
		size=function.size,
		signature=function.signature,
	)


def _symbol_size(symbol: Symbol) -> int:
	return symbol["st_size"]


def _functions_from_dwarf(dwarf: DWARFInfo | None) -> dict[Address, Function]:
	if dwarf is None:
		return {}  # pragma: no cover
	return {
		address: Function(
			name=_die_name(die),
			address=address,
			size=_subprogram_size(die),
			signature=_signature(_declaration(die)),
		)
		for compilation_unit in dwarf.iter_CUs()
		for die in _iter_dies(compilation_unit.get_top_DIE())
		if die.tag == "DW_TAG_subprogram"
		if (low_pc := die.attributes.get("DW_AT_low_pc")) is not None
		if (address := Address(_int_value(low_pc))) & ~1
	}


def _subprogram_size(die: DIE) -> int:
	low_pc = _int_value(die.attributes["DW_AT_low_pc"])
	high_pc = die.attributes.get("DW_AT_high_pc")
	if high_pc is None:
		return 0  # pragma: no cover
	if high_pc.form == "DW_FORM_addr":  # pragma: no branch
		return _int_value(high_pc) - low_pc  # pragma: no cover
	return _int_value(high_pc)


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
				FUNCTION_POINTER
				if pointee.tag == "DW_TAG_subroutine_type"
				else f"{_type_name(pointee)} *"
			)
		case "DW_TAG_typedef":
			return _type_name(_type_die(die))
		case "DW_TAG_structure_type" | "DW_TAG_union_type" | "DW_TAG_class_type":
			return layout_key(
				"union" if die.tag == "DW_TAG_union_type" else "struct", _die_name(die)
			)
		case "DW_TAG_array_type":
			return f"{_type_name(_type_die(die))}{ARRAY_SUFFIX}"
		case "DW_TAG_subroutine_type":  # pragma: no cover
			return FUNCTION_POINTER
		case _:
			return str(die.tag)  # pragma: no cover


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


class _FunctionPointer(Struct):
	signature: FunctionSignature


class _StructPointer(Struct):
	pointee: str | None


class _EmbeddedStruct(Struct):
	members: tuple[Member, ...]


type _PointeeKind = _FunctionPointer | _StructPointer | None
type _MemberKind = _FunctionPointer | _StructPointer | _EmbeddedStruct | None


def _pointee_kind(type_die: DIE | None) -> _PointeeKind:
	underlying = _strip_qualifiers(type_die)
	if underlying is None or underlying.tag != "DW_TAG_pointer_type":
		return None  # pragma: no cover
	pointee = _strip_qualifiers(_type_die(underlying))
	if pointee is None:
		return _StructPointer(pointee=None)
	match pointee.tag:
		case "DW_TAG_subroutine_type":
			return _FunctionPointer(signature=_signature(pointee))
		case "DW_TAG_structure_type" | "DW_TAG_union_type" if "DW_AT_name" in pointee.attributes:
			return _StructPointer(pointee=_type_name(pointee))
		case _:
			return None


def _member_kind(type_die: DIE | None) -> _MemberKind:
	underlying = _strip_qualifiers(type_die)
	if underlying is None:
		return None  # pragma: no cover
	if underlying.tag == "DW_TAG_pointer_type":
		return _pointee_kind(underlying)
	if underlying.tag in ("DW_TAG_structure_type", "DW_TAG_union_type"):
		return _EmbeddedStruct(members=tuple(_layout_members(underlying)))
	return None


def _layouts(dwarf: DWARFInfo | None) -> dict[str, StructureLayout]:
	if dwarf is None:
		return {}  # pragma: no cover
	layouts: dict[str, StructureLayout] = {}
	for compilation_unit in dwarf.iter_CUs():
		for die in _iter_dies(compilation_unit.get_top_DIE()):
			if die.tag not in ("DW_TAG_structure_type", "DW_TAG_union_type"):
				continue
			if "DW_AT_declaration" in die.attributes:
				continue
			name_attribute = die.attributes.get("DW_AT_name")
			if name_attribute is None:
				continue  # pragma: no cover
			name = _string_value(name_attribute)
			keyword = "union" if die.tag == "DW_TAG_union_type" else "struct"
			key = layout_key(keyword, name)
			byte_size = die.attributes.get("DW_AT_byte_size")
			layout = StructureLayout(
				members=tuple(_layout_members(die)),
				size=_int_value(byte_size) if byte_size is not None else 0,  # pragma: no branch
			)
			existing = layouts.get(key)
			if existing is None or len(layout.members) > len(existing.members):
				layouts[key] = layout
	return layouts


def _layout_members(struct_die: DIE) -> Iterator[Member]:
	for child in struct_die.iter_children():
		if child.tag != "DW_TAG_member":
			continue  # pragma: no cover
		offset = _member_offset(child)
		if offset is None:
			continue  # pragma: no cover
		match _member_kind(_type_die(child)):
			case _FunctionPointer(signature):
				yield FunctionPointerMember(
					kind="function_pointer",
					name=_member_name(child),
					offset=offset,
					signature=signature,
				)
			case _StructPointer(pointee):
				yield StructPointerMember(
					kind="struct_pointer",
					name=_member_name(child),
					offset=offset,
					pointee=pointee,
				)
			case _EmbeddedStruct(members):
				yield EmbeddedStructMember(
					kind="embedded_struct",
					name=_member_name(child),
					offset=offset,
					members=members,
				)
			case None:
				pass
			case _ as unreachable:
				assert_never(unreachable)


def _member_name(member_die: DIE) -> str | None:
	name_attribute = member_die.attributes.get("DW_AT_name")
	return None if name_attribute is None else _string_value(name_attribute)


def _member_offset(member_die: DIE) -> int | None:
	location = member_die.attributes.get("DW_AT_data_member_location")
	if location is None:
		return 0
	if location.form not in (
		"DW_FORM_data1",
		"DW_FORM_data2",
		"DW_FORM_data4",
		"DW_FORM_data8",
		"DW_FORM_udata",
		"DW_FORM_implicit_const",
	):
		return None  # pragma: no cover
	return _int_value(location)


def _objects_from_symtab(symtab: SymbolTableSection | None) -> dict[Address, DataObject]:
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


def _objects_from_dwarf(
	dwarf: DWARFInfo | None,
	pointer_size: int,
	byte_order: ByteOrder,
) -> dict[Address, DataObject]:
	if dwarf is None:
		return {}  # pragma: no cover
	return {
		address: DataObject(
			name=_string_value(name_attribute),
			address=address,
			size=_byte_size_of(die),
			type_name=_type_name(_type_die(die)),
			signature=_object_signature(die),
		)
		for compilation_unit in dwarf.iter_CUs()
		for die in _iter_dies(compilation_unit.get_top_DIE())
		if die.tag == "DW_TAG_variable"
		if "DW_AT_declaration" not in die.attributes
		if (name_attribute := die.attributes.get("DW_AT_name")) is not None
		if (address := _location_address(die, pointer_size, byte_order)) is not None
	}


def _object_signature(die: DIE) -> FunctionSignature | None:
	type_die = _type_die(die)
	if type_die is not None and type_die.tag == "DW_TAG_array_type":
		type_die = _type_die(type_die)
	match _pointee_kind(type_die):
		case _FunctionPointer(signature):
			return signature
		case _StructPointer() | None:
			return None
		case _ as unreachable:
			assert_never(unreachable)


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
	operations = _exprloc(location)
	if not operations or operations[0] != _DW_OP_ADDR:
		return None  # pragma: no cover
	return Address(int.from_bytes(bytes(operations[1 : 1 + pointer_size]), byte_order))


def _byte_size_of(die: DIE) -> int:
	type_die = _strip_qualifiers(_type_die(die))
	if type_die is None:
		return 0  # pragma: no cover
	byte_size = type_die.attributes.get("DW_AT_byte_size")
	return _int_value(byte_size) if byte_size is not None else 0  # pragma: no branch


def _declaration_types(dwarf: DWARFInfo | None) -> dict[str, str]:
	if dwarf is None:
		return {}  # pragma: no cover
	return {
		_string_value(name_attribute): _type_name(_type_die(die))
		for compilation_unit in dwarf.iter_CUs()
		for die in _iter_dies(compilation_unit.get_top_DIE())
		if die.tag == "DW_TAG_variable"
		if "DW_AT_declaration" in die.attributes
		if (name_attribute := die.attributes.get("DW_AT_name")) is not None
	}


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
		if data_object.size > merged[address].size:
			merged[address] = replace(merged[address], size=data_object.size)
	return merged


def _relocations(
	elf: ELFFile,
	sections: dict[str, _SectionBytes],
	pointer_size: int,
	byte_order: ByteOrder,
) -> tuple[Relocation, ...]:
	relocations: list[Relocation] = []
	for section in elf.iter_sections():
		if not isinstance(section, RelocationSection):
			continue
		target_index = section["sh_info"]
		target_section = elf.get_section(target_index)
		if target_index != 0 and (
			target_section is None or not target_section.header.sh_flags & _SHF_ALLOC
		):
			continue  # pragma: no branch
		symbol_table = elf.get_section(section["sh_link"])
		if not isinstance(symbol_table, SymbolTableSection):
			continue  # pragma: no cover
		for relocation in section.iter_relocations():
			slot = Address(relocation["r_offset"])
			addend = _relocation_addend(
				section.is_RELA(), relocation, sections, slot, pointer_size, byte_order
			)
			relocations.append(
				Relocation(
					slot=slot,
					target=Address(
						_relocation_target(
							relocation["r_info_sym"],
							symbol_table.get_symbol(relocation["r_info_sym"])["st_value"],
							addend,
						)
					),
					addend=addend,
					type_name=describe_reloc_type(relocation["r_info_type"], elf),
				)
			)
	return tuple(relocations)


def _relocation_addend(
	is_rela: bool,
	entry: ElfRelocation,
	sections: dict[str, _SectionBytes],
	slot: Address,
	pointer_size: int,
	byte_order: ByteOrder,
) -> int:
	if is_rela:
		return int(entry["r_addend"])
	target_section = next(
		(
			candidate
			for candidate in sections.values()
			if candidate.address <= slot < candidate.address + candidate.size
		),
		None,
	)
	if target_section is None:
		return 0
	offset = slot - target_section.address
	return int.from_bytes(target_section.data[offset : offset + pointer_size], byte_order)


def _relocation_target(symbol_index: int, symbol_value: int, addend: int) -> int:
	return addend if symbol_index == 0 else symbol_value + addend


def _iter_dies(root: DIE) -> Iterator[DIE]:
	yield root
	for child in root.iter_children():
		yield from _iter_dies(child)


def _int_value(attribute: AttributeValue) -> int:
	return int(cast("int | str | bytes", attribute.value))


def _string_value(attribute: AttributeValue) -> str:
	return _attr_string(cast("str | bytes", attribute.value))


def _exprloc(attribute: AttributeValue) -> list[int]:
	return cast("list[int]", attribute.value)


def _die_name(die: DIE) -> str:
	for attribute_name in ("DW_AT_linkage_name", "DW_AT_name"):
		attribute = die.attributes.get(attribute_name)
		if attribute is not None:
			return _string_value(attribute)
	origin = _origin(die)
	return ANONYMOUS if origin is None else _die_name(origin)


def _origin(die: DIE) -> DIE | None:
	return next(
		(
			die.get_DIE_from_attribute(attribute_name)
			for attribute_name in ("DW_AT_abstract_origin", "DW_AT_specification")
			if attribute_name in die.attributes
		),
		None,
	)


def _declaration(die: DIE) -> DIE:
	origin = _origin(die)
	return die if origin is None else _declaration(origin)


def _attr_string(value: str | bytes) -> str:
	return value if isinstance(value, str) else value.decode()  # pragma: no branch
