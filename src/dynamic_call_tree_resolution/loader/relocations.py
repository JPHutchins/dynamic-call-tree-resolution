# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Relocations: the ones the image applies at load, and the ones the linker kept."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, assert_never

from elftools.elf.descriptions import describe_reloc_type
from elftools.elf.relocation import Relocation as ElfRelocation
from elftools.elf.relocation import RelocationSection
from elftools.elf.sections import SymbolTableSection
from salix import Struct

from dynamic_call_tree_resolution.loader.sections import (
	SHF_ALLOC,
	SectionBytes,
)
from dynamic_call_tree_resolution.model import (
	Address,
	LinkReference,
	Machine,
	ReferenceKind,
	Relocation,
)

if TYPE_CHECKING:
	from collections.abc import Mapping

	from elftools.elf.elffile import ELFFile

	from dynamic_call_tree_resolution.model import ByteOrder


def relocated_sections(
	sections: dict[str, SectionBytes],
	relocations: tuple[Relocation, ...],
	pointer_size: int,
	byte_order: ByteOrder,
) -> dict[str, SectionBytes]:
	patches = {
		relocation.slot: relocation.target.to_bytes(pointer_size, byte_order)
		for relocation in relocations
		if relocation.target != 0
	}
	return {
		name: SectionBytes(
			address=section.address,
			size=section.size,
			data=_patched(section, patches, pointer_size),
			flags=section.flags,
		)
		for name, section in sections.items()
	}


def _patched(section: SectionBytes, patches: Mapping[Address, bytes], pointer_size: int) -> bytes:
	r"""The section's bytes with each relocated slot that fits inside it written.

	>>> section = SectionBytes(address=0x100, size=6, data=bytes(6), flags=0)
	>>> _patched(section, {Address(0x100): b"ABCD", Address(0x104): b"EFGH"}, 4)
	b'ABCD\x00\x00'
	"""
	data = bytearray(section.data)
	for slot, target in patches.items():
		if section.address <= slot and slot + pointer_size <= section.address + section.size:
			offset = slot - section.address
			data[offset : offset + pointer_size] = target
	return bytes(data)


def allocated_relocations(
	elf: ELFFile,
	sections: dict[str, SectionBytes],
	pointer_size: int,
	byte_order: ByteOrder,
) -> tuple[Relocation, ...]:
	relocations: list[Relocation] = []
	for section in elf.iter_sections():
		if not isinstance(section, RelocationSection) or not section.header.sh_flags & SHF_ALLOC:
			continue
		target_index = section["sh_info"]
		target_section = elf.get_section(target_index)
		if target_index != 0 and (
			target_section is None or not target_section.header.sh_flags & SHF_ALLOC
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
			symbol = symbol_table.get_symbol(relocation["r_info_sym"])
			relocations.append(
				Relocation(
					slot=slot,
					target=Address(
						_relocation_target(relocation["r_info_sym"], symbol["st_value"], addend)
					),
					addend=addend,
					type_name=describe_reloc_type(relocation["r_info_type"], elf),
					symbol=symbol.name
					if relocation["r_info_sym"] != 0 and symbol["st_shndx"] == "SHN_UNDEF"
					else None,
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


def link_references(elf: ELFFile, types: _ReferenceTypes) -> tuple[LinkReference, ...]:
	return tuple(
		LinkReference(
			slot=Address(relocation["r_offset"]),
			symbol="" if symbol["st_info"]["type"] == "STT_SECTION" else symbol.name,
			value=Address(symbol["st_value"]),
			kind=_reference_kind(types, relocation["r_info_type"]),
		)
		for section in elf.iter_sections()
		if isinstance(section, RelocationSection)
		and not section.header.sh_flags & SHF_ALLOC
		and (target := elf.get_section(section["sh_info"])) is not None
		and target.header.sh_flags & SHF_ALLOC
		and isinstance(symbols := elf.get_section(section["sh_link"]), SymbolTableSection)
		for relocation in section.iter_relocations()
		for symbol in (symbols.get_symbol(relocation["r_info_sym"]),)
	)


def reference_types(machine: Machine) -> _ReferenceTypes:
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
	sections: dict[str, SectionBytes],
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
