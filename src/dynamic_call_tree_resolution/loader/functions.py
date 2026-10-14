# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The image's functions, from its symbol table and DWARF, and what DWARF says of them."""

from __future__ import annotations

from itertools import groupby
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, assert_never

from elftools.dwarf.die import DIE
from elftools.elf.elffile import ELFFile

from dynamic_call_tree_resolution.loader.dwarf import (
	attr_string,
	declaration,
	die_name,
	die_origin,
	int_value,
	iter_dies,
	string_value,
)
from dynamic_call_tree_resolution.loader.layouts import (
	function_signature,
)
from dynamic_call_tree_resolution.loader.sections import (
	elf_symbol_table,
)
from dynamic_call_tree_resolution.model import (
	Address,
	Declaration,
	Function,
	InstructionFamily,
	Machine,
	SourceLocation,
	aligned,
	thumb_twin,
)

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path

	from elftools.dwarf.compileunit import CompileUnit
	from elftools.dwarf.dwarfinfo import DWARFInfo
	from elftools.elf.sections import Symbol, SymbolTableSection


def defined_function_names(path: Path) -> frozenset[str]:
	with path.open("rb") as stream:
		symtab = elf_symbol_table(ELFFile(stream))
		return frozenset(
			symbol.name
			for symbol in (symtab.iter_symbols() if symtab is not None else ())
			if symbol["st_info"]["type"] == "STT_FUNC"
			if symbol["st_shndx"] != "SHN_UNDEF"
		)


def functions_from_symtab(symtab: SymbolTableSection | None) -> dict[Address, Function]:
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


def merge_functions(
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


def functions_from_dwarf(dwarf: DWARFInfo | None) -> dict[Address, Function]:
	if dwarf is None:
		return {}
	return {
		address: Function(
			name=die_name(die),
			address=address,
			size=subprogram_size(die),
			signature=function_signature(declaration(die)),
		)
		for compilation_unit in dwarf.iter_CUs()
		for die in iter_dies(compilation_unit.get_top_DIE())
		if die.tag == "DW_TAG_subprogram"
		if (low_pc := die.attributes.get("DW_AT_low_pc")) is not None
		if (address := Address(int_value(low_pc))) & ~1
	}


def never_returns(dwarf: DWARFInfo | None, machine: Machine) -> frozenset[Address]:
	if dwarf is None:
		return frozenset()
	return frozenset(
		_code_start(Address(int_value(low_pc)), machine)
		for compilation_unit in dwarf.iter_CUs()
		for die in iter_dies(compilation_unit.get_top_DIE())
		if die.tag == "DW_TAG_subprogram"
		if (low_pc := die.attributes.get("DW_AT_low_pc")) is not None
		if int_value(low_pc) & ~1
		if _noreturn(die)
	)


def _noreturn(die: DIE) -> bool:
	origin = die_origin(die)
	return "DW_AT_noreturn" in die.attributes or (origin is not None and _noreturn(origin))


def parameterless(dwarf: DWARFInfo | None, machine: Machine) -> frozenset[Address]:
	if dwarf is None:
		return frozenset()
	return frozenset(
		_code_start(Address(int_value(low_pc)), machine)
		for compilation_unit in dwarf.iter_CUs()
		for die in iter_dies(compilation_unit.get_top_DIE())
		if die.tag == "DW_TAG_subprogram"
		if (low_pc := die.attributes.get("DW_AT_low_pc")) is not None
		if int_value(low_pc) & ~1
		if _prototyped(die) and not _declares_parameters(die)
	)


def _prototyped(die: DIE) -> bool:
	origin = die_origin(die)
	return bool(
		(prototyped := die.attributes.get("DW_AT_prototyped")) is not None and prototyped.value
	) or (origin is not None and _prototyped(origin))


def _declares_parameters(die: DIE) -> bool:
	origin = die_origin(die)
	return any(
		child.tag in ("DW_TAG_formal_parameter", "DW_TAG_unspecified_parameters")
		for child in die.iter_children()
	) or (origin is not None and _declares_parameters(origin))


def declarations(dwarf: DWARFInfo | None) -> dict[Address, Declaration]:
	if dwarf is None:
		return {}
	units = tuple(dwarf.iter_CUs())
	files = {unit.cu_offset: file_names(dwarf, unit) for unit in units}
	return {
		Address(int_value(low_pc)): Declaration(
			unit=string_value(unit.get_top_DIE().attributes["DW_AT_name"]),
			location=SourceLocation(
				file=file_name,
				line=int_value(declared.attributes["DW_AT_decl_line"]),
				column=int_value(column)
				if (column := declared.attributes.get("DW_AT_decl_column")) is not None
				else 0,
			),
		)
		for unit in units
		for die in iter_dies(unit.get_top_DIE())
		if die.tag == "DW_TAG_subprogram"
		if (low_pc := die.attributes.get("DW_AT_low_pc")) is not None
		if int_value(low_pc) & ~1
		for declared in (_declared(die),)
		if declared is not None
		for file_name in (
			files[declared.cu.cu_offset].get(int_value(declared.attributes["DW_AT_decl_file"])),
		)
		if file_name is not None
	}


def file_names(dwarf: DWARFInfo, unit: CompileUnit) -> dict[int, str]:
	program = dwarf.line_program_for_CU(unit)
	return (
		{
			index + (0 if program["version"] >= 5 else 1): PurePosixPath(
				attr_string(entry.name)
			).name
			for index, entry in enumerate(program["file_entry"])
		}
		if program is not None
		else {}
	)


def _declared(die: DIE) -> DIE | None:
	match die.attributes.get("DW_AT_decl_line"), die_origin(die):
		case None, None:
			return None  # pragma: no cover
		case None, DIE() as origin:
			return _declared(origin)
		case _:
			return die


def symbol_addresses(
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


def labels(symtab: SymbolTableSection | None) -> dict[str, Address]:
	return {
		name: next(iter(addresses))
		for name, group in groupby(
			sorted(
				(symbol.name, Address(symbol["st_value"]))
				for symbol in (symtab.iter_symbols() if symtab is not None else ())
				if symbol["st_info"]["type"] == "STT_NOTYPE"
				and symbol["st_shndx"] != "SHN_UNDEF"
				and symbol.name
				and not symbol.name.startswith("$")
			),
			key=_symbol_name,
		)
		for addresses in ({address for _, address in group},)
		if len(addresses) == 1
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


def subprogram_size(die: DIE) -> int:
	low_pc = int_value(die.attributes["DW_AT_low_pc"])
	high_pc = die.attributes.get("DW_AT_high_pc")
	if high_pc is None:
		return 0  # pragma: no cover
	if high_pc.form == "DW_FORM_addr":
		return int_value(high_pc) - low_pc
	return int_value(high_pc)
