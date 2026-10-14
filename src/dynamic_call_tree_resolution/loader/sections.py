# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The image's allocated sections, mapping symbols and segments."""

from __future__ import annotations

import re
from itertools import groupby
from typing import TYPE_CHECKING, Final

from elftools.elf.sections import ARMAttributesSection, SymbolTableSection
from salix import Struct

from dynamic_call_tree_resolution.model import (
	Address,
	ArmCore,
	ArmProfile,
)

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator

	from elftools.elf.elffile import ELFFile
	from elftools.elf.sections import Symbol

	from dynamic_call_tree_resolution.model import ByteOrder


class SectionBytes(Struct):
	"""One section's address, size, bytes and flags, as the ELF holds them."""

	address: int
	size: int
	data: bytes
	flags: int


SHF_WRITE: Final = 0x1


SHF_ALLOC: Final = 0x2


SHF_EXECINSTR: Final = 0x4


SHF_TLS: Final = 0x400


_MAPPING_SYMBOL: Final = re.compile(r"\$[adt](\..*)?")


def aligned_total(blocks: Iterable[tuple[int, int]]) -> int:
	"""Each block's size rounded up to its alignment, summed.

	>>> aligned_total(((4, 8), (4, 4)))
	12
	>>> aligned_total(((4, 0), (6, 1)))
	10
	"""
	return sum(
		size if alignment <= 1 else (size + alignment - 1) // alignment * alignment
		for size, alignment in blocks
	)


def arm_core(elf: ELFFile) -> ArmCore | None:
	match elf.get_section_by_name(".ARM.attributes"):
		case ARMAttributesSection() as section:
			attributes = {
				attribute.tag: attribute.value
				for subsection in section.iter_subsections()
				for subsubsection in subsection.iter_subsubsections()
				for attribute in subsubsection.iter_attributes()
			}
			profile = attributes.get("TAG_CPU_ARCH_PROFILE")
			return (
				ArmCore(
					profile=ArmProfile(chr(profile)),
					floating_point=bool(attributes.get("TAG_FP_ARCH")),
				)
				if isinstance(profile, int) and chr(profile) in ArmProfile
				else None
			)
		case _:
			return None


def mapped_spans(
	elf: ELFFile, markers: tuple[tuple[int, int, str], ...], kind: str
) -> tuple[tuple[Address, Address], ...]:
	return tuple(
		span
		for section_index, group in groupby(markers, key=_marker_section)
		if (section := elf.get_section(section_index)) is not None
		and section.header.sh_flags & SHF_EXECINSTR
		for span in _kind_spans(tuple(group), section.header.sh_addr + section.header.sh_size, kind)
	)


def mapping_markers(symtab: SymbolTableSection | None) -> tuple[tuple[int, int, str], ...]:
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


def load_address(elf: ELFFile, address: int) -> Address | None:
	return next(
		(
			Address(segment["p_paddr"] + address - segment["p_vaddr"])
			for segment in elf.iter_segments(type="PT_LOAD")
			if segment["p_vaddr"] <= address < segment["p_vaddr"] + segment["p_memsz"]
			and segment["p_paddr"] != segment["p_vaddr"]
		),
		None,
	)


def relro(spans: tuple[tuple[int, int], ...], section: SectionBytes) -> bool:
	return any(
		start <= section.address and section.address + section.size <= end for start, end in spans
	)


def elf_byte_order(elf: ELFFile) -> ByteOrder:
	return "little" if elf.little_endian else "big"


def elf_symbol_table(elf: ELFFile) -> SymbolTableSection | None:
	section = elf.get_section_by_name(".symtab")
	return section if isinstance(section, SymbolTableSection) else None  # pragma: no branch


def elf_sections(elf: ELFFile) -> dict[str, SectionBytes]:
	return {
		section.name: SectionBytes(
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
