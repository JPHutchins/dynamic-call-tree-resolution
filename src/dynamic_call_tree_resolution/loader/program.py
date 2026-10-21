# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Assembling the :class:`Program` model from an ELF image."""

from __future__ import annotations

from typing import TYPE_CHECKING

from elftools.elf.elffile import ELFFile

from dynamic_call_tree_resolution.loader.functions import (
	declarations,
	functions_from_dwarf,
	functions_from_symtab,
	labels,
	merge_functions,
	never_returns,
	parameterless,
	symbol_addresses,
)
from dynamic_call_tree_resolution.loader.layouts import (
	structure_layouts,
)
from dynamic_call_tree_resolution.loader.objects import (
	merge_objects,
	objects_from_dwarf,
	objects_from_symtab,
	types_of_declarations,
)
from dynamic_call_tree_resolution.loader.relocations import (
	allocated_relocations,
	link_references,
	reference_types,
	relocated_sections,
)
from dynamic_call_tree_resolution.loader.sections import (
	SHF_ALLOC,
	SHF_EXECINSTR,
	SHF_TLS,
	SHF_WRITE,
	AlignedBlock,
	aligned_total,
	arm_core,
	elf_byte_order,
	elf_sections,
	elf_symbol_table,
	load_address,
	mapped_spans,
	mapping_markers,
	relro,
)
from dynamic_call_tree_resolution.loader.spans import (
	inlined,
)
from dynamic_call_tree_resolution.model import (
	Address,
	Machine,
	Program,
	Section,
)
from dynamic_call_tree_resolution.span import Span

if TYPE_CHECKING:
	from pathlib import Path
	from typing import BinaryIO


def load(path: Path) -> Program:
	with path.open("rb") as stream:
		return _load(stream)


def elf_machine(path: Path) -> str:
	with path.open("rb") as stream:
		return ELFFile(stream).header["e_machine"]


def _load(stream: BinaryIO) -> Program:
	elf = ELFFile(stream)
	if elf.header["e_type"] == "ET_REL":
		raise ValueError("relocatable (ET_REL) images are not supported; link the image first")
	dwarf = elf.get_dwarf_info(relocate_dwarf_sections=False) if elf.has_dwarf_info() else None
	symtab = elf_symbol_table(elf)
	pointer_size = elf.elfclass // 8
	byte_order = elf_byte_order(elf)
	raw_sections = elf_sections(elf)
	relocations = allocated_relocations(elf, raw_sections, pointer_size, byte_order)
	sections = relocated_sections(raw_sections, relocations, pointer_size, byte_order)
	relro_spans = tuple(
		Span(
			start=Address(segment["p_vaddr"]),
			end=Address(segment["p_vaddr"] + segment["p_memsz"]),
		)
		for segment in elf.iter_segments(type="PT_GNU_RELRO")
	)
	machine = Machine(elf.header["e_machine"])
	markers = mapping_markers(symtab)
	return Program(
		byte_order=byte_order,
		pointer_size=pointer_size,
		machine=machine,
		functions=merge_functions(
			functions_from_symtab(symtab), functions_from_dwarf(dwarf), machine
		),
		objects=merge_objects(
			objects_from_symtab(symtab),
			objects_from_dwarf(dwarf, pointer_size, byte_order),
			types_of_declarations(dwarf),
		),
		layouts=structure_layouts(dwarf),
		relocations=relocations,
		sections={
			Address(section.address): Section(
				data=section.data,
				writable=bool(section.flags & SHF_WRITE) and not relro(relro_spans, section),
				executable=bool(section.flags & SHF_EXECINSTR),
				load_address=load_address(elf, section.address),
			)
			for section in sections.values()
			if section.flags & SHF_ALLOC
		},
		data_in_code=mapped_spans(elf, markers, "d"),
		arm_code=tuple(sorted(mapped_spans(elf, markers, "a"))),
		link_references=link_references(elf, reference_types(machine)),
		inlined=inlined(dwarf),
		declarations=declarations(dwarf),
		never_returns=never_returns(dwarf, machine),
		parameterless=parameterless(dwarf, machine),
		symbol_addresses=symbol_addresses(symtab, machine),
		labels=labels(symtab),
		arm_core=arm_core(elf),
		tls_size=aligned_total(
			AlignedBlock(size=section.header.sh_size, alignment=section.header.sh_addralign)
			for section in elf.iter_sections()
			if section.header.sh_flags & SHF_TLS
		),
		entry_point=Address(elf.header["e_entry"]),
		nobits=tuple(
			sorted(
				Span(
					start=Address(section.header.sh_addr),
					end=Address(section.header.sh_addr + section.header.sh_size),
				)
				for section in elf.iter_sections()
				if section.header.sh_type == "SHT_NOBITS"
				and section.header.sh_flags & SHF_ALLOC
				and section.header.sh_size > 0
				and not section.header.sh_flags & SHF_TLS
			)
		),
	)
