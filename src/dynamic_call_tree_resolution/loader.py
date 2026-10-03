# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Loading ELF and DWARF data into the immutable :class:`Program` model."""

from __future__ import annotations

import re
from enum import IntEnum
from itertools import accumulate, groupby
from math import prod
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Final, assert_never, cast

from elftools.dwarf.die import DIE, AttributeValue
from elftools.dwarf.ranges import BaseAddressEntry, RangeEntry
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
	ArrayMember,
	DataObject,
	Declaration,
	EmbeddedStructMember,
	Function,
	FunctionPointerMember,
	FunctionSignature,
	InstructionFamily,
	LinkReference,
	Machine,
	Program,
	ReferenceKind,
	Relocation,
	Section,
	SkippedMember,
	SkipReason,
	SourceLocation,
	StructPointerMember,
	StructureLayout,
	aligned,
	layout_key,
	thumb_twin,
)

if TYPE_CHECKING:
	from collections.abc import Callable, Iterator, Mapping
	from pathlib import Path
	from typing import BinaryIO

	from elftools.dwarf.compileunit import CompileUnit
	from elftools.dwarf.dwarfinfo import DWARFInfo
	from elftools.dwarf.ranges import RangeLists
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
_CONSTANT_FORMS: Final = frozenset(
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


def load(path: Path) -> Program:
	with path.open("rb") as stream:
		return _load(stream)


def elf_machine(path: Path) -> str:
	with path.open("rb") as stream:
		return ELFFile(stream).header["e_machine"]


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
	dwarf = elf.get_dwarf_info(relocate_dwarf_sections=False) if elf.has_dwarf_info() else None
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
		link_references=_link_references(elf, _reference_types(machine)),
		inlined=_inlined(dwarf),
		declarations=_declarations(dwarf),
		symbol_addresses=_symbol_addresses(symtab, machine),
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
	address = _symbol_address(function, from_symtab, machine)
	symbol = from_symtab.get(address)
	return Function(
		name=symbol.name if symbol is not None else function.name,
		address=address,
		size=function.size,
		signature=function.signature,
	)


def _symbol_address(
	function: Function, from_symtab: Mapping[Address, Function], machine: Machine
) -> Address:
	match machine.family:
		case InstructionFamily.ARM if thumb_twin(function.address) in from_symtab:
			return thumb_twin(function.address)
		case InstructionFamily.ARM | InstructionFamily.X86:
			return function.address
		case _ as unreachable:
			assert_never(unreachable)


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


def _declarations(dwarf: DWARFInfo | None) -> dict[Address, Declaration]:
	if dwarf is None:
		return {}  # pragma: no cover
	units = tuple(dwarf.iter_CUs())
	files = {unit.cu_offset: _file_names(dwarf, unit) for unit in units}
	return {
		Address(_int_value(low_pc)): Declaration(
			unit=_string_value(unit.get_top_DIE().attributes["DW_AT_name"]),
			location=SourceLocation(
				file=file_name,
				line=_int_value(declared.attributes["DW_AT_decl_line"]),
				column=_int_value(column)
				if (column := declared.attributes.get("DW_AT_decl_column")) is not None
				else 0,
			),
		)
		for unit in units
		for die in _iter_dies(unit.get_top_DIE())
		if die.tag == "DW_TAG_subprogram"
		if (low_pc := die.attributes.get("DW_AT_low_pc")) is not None
		if _int_value(low_pc) & ~1
		for declared in (_declared(die),)
		if declared is not None
		for file_name in (
			files[declared.cu.cu_offset].get(_int_value(declared.attributes["DW_AT_decl_file"])),
		)
		if file_name is not None
	}


def _file_names(dwarf: DWARFInfo, unit: CompileUnit) -> dict[int, str]:
	program = dwarf.line_program_for_CU(unit)
	return (
		{
			index + (0 if program["version"] >= 5 else 1): PurePosixPath(
				_attr_string(entry.name)
			).name
			for index, entry in enumerate(program["file_entry"])
		}
		if program is not None
		else {}
	)


def _declared(die: DIE) -> DIE | None:
	match die.attributes.get("DW_AT_decl_line"), _origin(die):
		case None, None:
			return None  # pragma: no cover
		case None, DIE() as origin:
			return _declared(origin)
		case _:
			return die


def _symbol_addresses(
	symtab: SymbolTableSection | None, machine: Machine
) -> dict[str, frozenset[Address]]:
	if symtab is None:
		return {}  # pragma: no cover
	named = sorted(
		(symbol.name, _code_start(Address(symbol["st_value"]), machine))
		for symbol in symtab.iter_symbols()
		if symbol["st_info"]["type"] == "STT_FUNC"
		if symbol["st_shndx"] != "SHN_UNDEF"
	)
	return {
		name: frozenset(address for _, address in group)
		for name, group in groupby(named, key=_symbol_name)
	}


def _symbol_name(named: tuple[str, Address]) -> str:
	return named[0]


def _code_start(address: Address, machine: Machine) -> Address:
	match machine.family:
		case InstructionFamily.ARM:
			return aligned(address)
		case InstructionFamily.X86:
			return address
		case _ as unreachable:
			assert_never(unreachable)


def _inlined(dwarf: DWARFInfo | None) -> dict[str, tuple[tuple[Address, Address], ...]]:
	if dwarf is None:
		return {}  # pragma: no cover
	dies = tuple(
		(unit, die)
		for unit in dwarf.iter_CUs()
		for die in _iter_dies(unit.get_top_DIE())
		if die.tag in ("DW_TAG_subprogram", "DW_TAG_inlined_subroutine")
	)
	abstract = {
		die.offset: _die_name(die)
		for _, die in dies
		if die.tag == "DW_TAG_subprogram" and "DW_AT_inline" in die.attributes
	}
	return {
		name: tuple(span for _, span in group)
		for name, group in groupby(
			sorted(
				(abstract[origin], span)
				for unit, die in dies
				if die.tag == "DW_TAG_inlined_subroutine"
				for origin in (_reference(unit, die.attributes.get("DW_AT_abstract_origin")),)
				if origin in abstract
				for span in _die_spans(dwarf, unit, die)
			),
			key=_span_name,
		)
	}


def _span_name(named_span: tuple[str, tuple[Address, Address]]) -> str:
	return named_span[0]


def _reference(unit: CompileUnit, attribute: AttributeValue | None) -> int | None:
	match attribute:
		case None:
			return None  # pragma: no cover
		case AttributeValue(form="DW_FORM_ref_addr"):
			return _int_value(attribute)  # pragma: no cover
		case AttributeValue():
			return unit.cu_offset + _int_value(attribute)
		case _ as unreachable:
			assert_never(unreachable)


def _die_spans(
	dwarf: DWARFInfo, unit: CompileUnit, die: DIE
) -> tuple[tuple[Address, Address], ...]:
	match die.attributes.get("DW_AT_ranges"), die.attributes.get("DW_AT_low_pc"):
		case AttributeValue() as ranges, _:
			return _range_spans(dwarf.range_lists(), _int_value(ranges), unit)
		case None, AttributeValue() as low_pc:
			return (
				(
					Address(_int_value(low_pc)),
					Address(_int_value(low_pc) + _subprogram_size(die)),
				),
			)
		case _:
			return ()  # pragma: no cover


def _range_spans(
	range_lists: RangeLists | None, offset: int, unit: CompileUnit
) -> tuple[tuple[Address, Address], ...]:
	if range_lists is None:
		return ()  # pragma: no cover
	entries = range_lists.get_range_list_at_offset(offset, cu=unit)
	return tuple(
		(Address(begin), Address(end))
		for entry, base in zip(
			entries, accumulate(entries, _range_base, initial=_unit_base(unit)), strict=False
		)
		if isinstance(entry, RangeEntry)
		for begin, end in (
			(entry.begin_offset, entry.end_offset)
			if entry.is_absolute
			else (base + entry.begin_offset, base + entry.end_offset),
		)
	)


def _range_base(base: int, entry: RangeEntry | BaseAddressEntry) -> int:
	return entry.base_address if isinstance(entry, BaseAddressEntry) else base


def _unit_base(unit: CompileUnit) -> int:
	low_pc = unit.get_top_DIE().attributes.get("DW_AT_low_pc")
	return _int_value(low_pc) if low_pc is not None else 0


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
	return _named_type(die, _shared_anonymous)


def _layout_name(die: DIE | None) -> str:
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
			return _die_name(die)
		case "DW_TAG_const_type" | "DW_TAG_volatile_type":
			return _named_type(_type_die(die), anonymous)
		case "DW_TAG_pointer_type":
			pointee = _strip_qualifiers(_type_die(die))
			if pointee is None:
				return "void *"
			return (
				FUNCTION_POINTER
				if pointee.tag == "DW_TAG_subroutine_type"
				else f"{_named_type(_type_die(die), anonymous)} *"
			)
		case "DW_TAG_typedef":
			return (
				_die_name(die)
				if _anonymous_aggregate(_type_die(die))
				else _named_type(_type_die(die), anonymous)
			)
		case "DW_TAG_structure_type" | "DW_TAG_union_type" | "DW_TAG_class_type":
			return layout_key(
				"union" if die.tag == "DW_TAG_union_type" else "struct",
				_die_name(die) if "DW_AT_name" in die.attributes else anonymous(die),
			)
		case "DW_TAG_array_type":
			return f"{_named_type(_type_die(die), anonymous)}{ARRAY_SUFFIX}"
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


class _Array(Struct):
	count: int
	stride: int
	element: Member


class _Skipped(Struct):
	reason: SkipReason


type _PointeeKind = _FunctionPointer | _StructPointer | _Skipped | None
type _MemberKind = _FunctionPointer | _StructPointer | _EmbeddedStruct | _Array | _Skipped | None


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
		case "DW_TAG_structure_type" | "DW_TAG_union_type":
			return _StructPointer(pointee=_layout_name(_type_die(underlying)))
		case "DW_TAG_pointer_type" if _pointee_kind(pointee) is not None:
			return _Skipped(reason=SkipReason.POINTER_TO_POINTER)
		case "DW_TAG_array_type" if _array_kind(pointee) is not None:
			return _Skipped(reason=SkipReason.POINTER_TO_ARRAY)
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
	if underlying.tag == "DW_TAG_array_type":
		return _array_kind(underlying)
	return None


def _array_kind(array_die: DIE) -> _Array | _Skipped | None:
	element = _member(_member_kind(_type_die(array_die)), None, 0)
	if element is None:
		return None
	stride = _byte_size(_strip_qualifiers(_type_die(array_die)))
	counts = _known_counts(array_die)
	if not stride or counts is None:
		return _Skipped(reason=SkipReason.UNSIZED_ARRAY)
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
		_int_value(attribute)
		if attribute is not None and attribute.form in _CONSTANT_FORMS
		else None
	)


def _byte_size(die: DIE | None) -> int | None:
	return _constant(die.attributes.get("DW_AT_byte_size")) if die is not None else None


def _member(kind: _MemberKind, name: str | None, offset: int) -> Member | None:
	match kind:
		case _FunctionPointer(signature):
			return FunctionPointerMember(
				kind="function_pointer", name=name, offset=offset, signature=signature
			)
		case _StructPointer(pointee):
			return StructPointerMember(
				kind="struct_pointer", name=name, offset=offset, pointee=pointee
			)
		case _EmbeddedStruct(members):
			return EmbeddedStructMember(
				kind="embedded_struct", name=name, offset=offset, members=members
			)
		case _Array() as array:
			return _array_member(array, name, offset)
		case _Skipped(reason):
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


def _layouts(dwarf: DWARFInfo | None) -> dict[str, StructureLayout]:
	if dwarf is None:
		return {}  # pragma: no cover
	layouts: dict[str, StructureLayout] = {}
	for compilation_unit in dwarf.iter_CUs():
		for die in _iter_dies(compilation_unit.get_top_DIE()):
			aggregate = _defined_aggregate(die)
			if aggregate is None:
				continue
			key = _layout_name(die)
			layout = StructureLayout(
				members=tuple(_layout_members(aggregate)), size=_byte_size(aggregate) or 0
			)
			existing = layouts.get(key)
			if existing is None or len(layout.members) > len(existing.members):
				layouts[key] = layout
	return layouts


def _defined_aggregate(die: DIE) -> DIE | None:
	aggregate = (
		die
		if die.tag in ("DW_TAG_structure_type", "DW_TAG_union_type")
		else _type_die(die)
		if die.tag == "DW_TAG_typedef" and _anonymous_aggregate(_type_die(die))
		else None
	)
	return (
		aggregate
		if aggregate is not None and "DW_AT_declaration" not in aggregate.attributes
		else None
	)


def _layout_members(struct_die: DIE) -> Iterator[Member]:
	for child in struct_die.iter_children():
		if child.tag != "DW_TAG_member":
			continue  # pragma: no cover
		offset = _member_offset(child)
		if offset is None:
			continue  # pragma: no cover
		member = _member(_member_kind(_type_die(child)), _member_name(child), offset)
		if member is not None:
			yield member


def _member_name(member_die: DIE) -> str | None:
	name_attribute = member_die.attributes.get("DW_AT_name")
	return None if name_attribute is None else _string_value(name_attribute)


def _member_offset(member_die: DIE) -> int | None:
	location = member_die.attributes.get("DW_AT_data_member_location")
	if location is None:
		return 0
	if location.form not in _CONSTANT_FORMS:
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
			type_name=_layout_name(_type_die(die)),
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
		case _StructPointer() | _Skipped() | None:
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
		_string_value(name_attribute): _layout_name(_type_die(die))
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
		if not isinstance(section, RelocationSection) or not section.header.sh_flags & _SHF_ALLOC:
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


class ArmRelocation(IntEnum):
	"""The ARM relocation types that dctr classifies, by their ELF ABI numbers."""

	R_ARM_ABS32 = 2
	R_ARM_REL32 = 3
	R_ARM_THM_CALL = 10
	R_ARM_CALL = 28
	R_ARM_JUMP24 = 29
	R_ARM_THM_JUMP24 = 30
	R_ARM_MOVW_ABS_NC = 43
	R_ARM_MOVT_ABS = 44
	R_ARM_THM_MOVW_ABS_NC = 47
	R_ARM_THM_MOVT_ABS = 48
	R_ARM_THM_JUMP19 = 51


class X64Relocation(IntEnum):
	"""The x86-64 relocation types that dctr classifies, by their ELF ABI numbers."""

	R_X86_64_64 = 1
	R_X86_64_PC32 = 2
	R_X86_64_PLT32 = 4
	R_X86_64_32 = 10
	R_X86_64_32S = 11


class I386Relocation(IntEnum):
	"""The i386 relocation types that dctr classifies, by their ELF ABI numbers."""

	R_386_32 = 1
	R_386_PC32 = 2
	R_386_PLT32 = 4


class _ReferenceTypes(Struct):
	"""One machine's relocation types, by what they do with their symbol."""

	addresses: frozenset[int]
	calls: frozenset[int]


def _link_references(elf: ELFFile, types: _ReferenceTypes) -> tuple[LinkReference, ...]:
	return tuple(
		LinkReference(
			slot=Address(relocation["r_offset"]),
			symbol="" if symbol["st_info"]["type"] == "STT_SECTION" else symbol.name,
			value=Address(symbol["st_value"]),
			kind=_reference_kind(types, relocation["r_info_type"]),
		)
		for section in elf.iter_sections()
		if isinstance(section, RelocationSection)
		and not section.header.sh_flags & _SHF_ALLOC
		and (target := elf.get_section(section["sh_info"])) is not None
		and target.header.sh_flags & _SHF_ALLOC
		and isinstance(symbols := elf.get_section(section["sh_link"]), SymbolTableSection)
		for relocation in section.iter_relocations()
		for symbol in (symbols.get_symbol(relocation["r_info_sym"]),)
	)


def _reference_types(machine: Machine) -> _ReferenceTypes:
	match machine:
		case Machine.EM_ARM:
			return _ReferenceTypes(
				addresses=frozenset(
					{
						ArmRelocation.R_ARM_ABS32,
						ArmRelocation.R_ARM_REL32,
						ArmRelocation.R_ARM_MOVW_ABS_NC,
						ArmRelocation.R_ARM_MOVT_ABS,
						ArmRelocation.R_ARM_THM_MOVW_ABS_NC,
						ArmRelocation.R_ARM_THM_MOVT_ABS,
					}
				),
				calls=frozenset(
					{
						ArmRelocation.R_ARM_THM_CALL,
						ArmRelocation.R_ARM_THM_JUMP24,
						ArmRelocation.R_ARM_THM_JUMP19,
						ArmRelocation.R_ARM_CALL,
						ArmRelocation.R_ARM_JUMP24,
					}
				),
			)
		case Machine.EM_X86_64:
			return _ReferenceTypes(
				addresses=frozenset(
					{
						X64Relocation.R_X86_64_64,
						X64Relocation.R_X86_64_32,
						X64Relocation.R_X86_64_32S,
						X64Relocation.R_X86_64_PC32,
					}
				),
				calls=frozenset({X64Relocation.R_X86_64_PLT32}),
			)
		case Machine.EM_386:
			return _ReferenceTypes(
				addresses=frozenset({I386Relocation.R_386_32}),
				calls=frozenset({I386Relocation.R_386_PC32, I386Relocation.R_386_PLT32}),
			)
		case _ as unreachable:
			assert_never(unreachable)


def _reference_kind(types: _ReferenceTypes, relocation_type: int) -> ReferenceKind:
	return (
		ReferenceKind.ADDRESS
		if relocation_type in types.addresses
		else ReferenceKind.CALL
		if relocation_type in types.calls
		else ReferenceKind.OTHER
	)


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
