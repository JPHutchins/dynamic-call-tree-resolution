# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""A build's C units, each with the alias dump GCC wrote beside its object and its object's types."""

from __future__ import annotations

import re
import shlex
from glob import escape
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Final

import msgspec
from elftools.elf.elffile import ELFFile
from salix import Struct

from dynamic_call_tree_resolution.composition import flatmap, merged
from dynamic_call_tree_resolution.loader.dwarf import die_name, iter_dies
from dynamic_call_tree_resolution.loader.layouts import (
	die_type,
	layout_name,
	strip_qualifiers,
	structure_layouts,
)
from dynamic_call_tree_resolution.loader.sections import elf_symbol_table
from dynamic_call_tree_resolution.model import ANONYMOUS

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator, Mapping

	from elftools.dwarf.die import DIE
	from elftools.elf.sections import Symbol, SymbolTableSection

	from dynamic_call_tree_resolution.model import StructureLayout

DUMP_FLAG: Final = "-fdump-tree-alias-lineno"
_NUMBERED: Final = re.compile(r"^([A-Za-z_]\w*)\.\d+$")


class Record(Struct):
	"""A structure or union: the name the plugin prints, and its layout's key."""

	name: str
	key: str


class UnitTypes(Struct):
	"""What one unit's own debug information says about the names its dump uses."""

	layouts: Mapping[str, StructureLayout]
	typedefs: Mapping[str, Record]
	"""Each typedef of a structure or union, by the typedef's name."""
	variables: Mapping[str, str]
	"""Each file-scope variable's type, as ``DataObject.type_name`` names it."""
	functions: frozenset[str]
	statics: Mapping[str, str]
	"""The symbol of each function-scope static whose name no other one in the unit shares."""


class DumpedUnit(Struct):
	"""One C unit of the build, with its alias dump when GCC wrote one."""

	unit: str
	"""The build's path of the unit's object, without its suffix."""
	source: Path
	object: Path
	dump: Path | None
	"""None for a unit without a function, for which GCC writes no dump."""


class _CompileCommand(Struct):
	"""One entry of a CMake compile_commands.json."""

	directory: str
	command: str
	file: str
	output: str


class _SymbolAddress(Struct, order=True):
	"""Where a symbol is in its object."""

	section: str
	address: int


class _FunctionSymbol(Struct, order=True):
	"""A defined function's symbol: a function's aliases share its address."""

	at: _SymbolAddress
	name: str


class _NumberedSymbol(Struct, order=True):
	"""A function-scope static's symbol, as GCC numbers it: ``hook.0`` for ``hook``."""

	name: str
	symbol: str


def dumped_units(build_directory: Path) -> tuple[DumpedUnit, ...]:
	"""The build's C units, each paired with its alias dump.

	Raises:
		ValueError: when a unit was compiled without the dump flag, when its dump is older than its
			source, so that it was written before the source last changed, or when several dumps
			match its object.
	"""
	commands = tuple(
		_CompileCommand(
			directory=entry["directory"],
			command=entry["command"],
			file=entry["file"],
			output=entry["output"],
		)
		for entry in msgspec.json.decode(
			(build_directory / "compile_commands.json").read_bytes(),
			type=tuple[dict[str, str], ...],
		)
		if entry["file"].endswith(".c")
	)
	undumped = tuple(
		command.file for command in commands if DUMP_FLAG not in shlex.split(command.command)
	)
	if undumped:
		raise ValueError(
			f"{len(undumped)} of {len(commands)} C units were compiled without {DUMP_FLAG}, such "
			f"as {undumped[0]}"
		)
	units = tuple(_dumped_unit(build_directory, command) for command in commands)
	stale = tuple(
		unit.unit
		for unit in units
		if unit.dump is not None
		and unit.source.exists()
		and unit.dump.stat().st_mtime < unit.source.stat().st_mtime
	)
	if stale:
		raise ValueError(
			f"{len(stale)} alias dumps are older than their sources, such as {stale[0]}'s: rebuild"
		)
	return units


def _dumped_unit(build_directory: Path, command: _CompileCommand) -> DumpedUnit:
	"""One unit, by its object's path in the build, or in its entry's directory if the build moved.

	Raises:
		ValueError: when several dumps match the unit's object.
	"""
	recorded = Path(command.directory) / command.output
	output = (
		recorded.relative_to(build_directory.resolve())
		if recorded.is_relative_to(build_directory.resolve())
		else recorded.relative_to(command.directory)
	)
	dumps = tuple(
		sorted(
			(build_directory / output.parent).glob(
				f"{escape(output.stem)}{escape(Path(command.file).suffix)}.*.alias"
			)
		)
	)
	if len(dumps) > 1:
		raise ValueError(
			f"{len(dumps)} alias dumps match {output}, such as {dumps[0]}: remove all but one"
		)
	return DumpedUnit(
		unit=output.with_suffix("").as_posix(),
		source=Path(command.directory) / command.file,
		object=build_directory / output,
		dump=dumps[0] if dumps else None,
	)


def defined_functions(object_path: Path) -> tuple[frozenset[str], ...]:
	"""The functions an object defines, each as the names of the symbols at its address."""
	with object_path.open("rb") as stream:
		return tuple(
			frozenset(symbol.name for symbol in group)
			for _, group in groupby(
				sorted(
					_FunctionSymbol(
						at=_SymbolAddress(
							section=str(symbol["st_shndx"]), address=symbol["st_value"]
						),
						name=symbol.name,
					)
					for symbol in _symbols(elf_symbol_table(ELFFile(stream)))
					if symbol["st_info"]["type"] == "STT_FUNC" and symbol["st_shndx"] != "SHN_UNDEF"
				),
				key=_symbol_address,
			)
		)


def _symbol_address(symbol: _FunctionSymbol) -> _SymbolAddress:
	return symbol.at


def _symbols(symtab: SymbolTableSection | None) -> Iterable[Symbol]:
	return symtab.iter_symbols() if symtab is not None else ()


def unit_types(object_path: Path) -> UnitTypes:
	"""What one unit's object says about the types its dump names.

	Raises:
		ValueError: when the object has no debug information to name its types.
	"""
	with object_path.open("rb") as stream:
		elf = ELFFile(stream)
		if not elf.has_dwarf_info():
			raise ValueError(f"{object_path} has no debug information: build with -g")
		dwarf = elf.get_dwarf_info(
			relocate_dwarf_sections=elf.get_section_by_name(".rela.debug_info") is not None
		)
		dies = tuple(flatmap(lambda unit: iter_dies(unit.get_top_DIE()), dwarf.iter_CUs()))
		return UnitTypes(
			layouts=structure_layouts(dwarf),
			typedefs=merged(map(_typedef, dies)),
			variables={
				die_name(die): layout_name(die_type(die))
				for die in reversed(
					tuple(
						flatmap(lambda unit: unit.get_top_DIE().iter_children(), dwarf.iter_CUs())
					)
				)
				if die.tag == "DW_TAG_variable" and "DW_AT_name" in die.attributes
			},
			functions=frozenset(
				die_name(die)
				for die in dies
				if die.tag == "DW_TAG_subprogram" and "DW_AT_name" in die.attributes
			),
			statics=_statics(elf_symbol_table(elf)),
		)


def _statics(symtab: SymbolTableSection | None) -> Mapping[str, str]:
	"""Each function-scope static's symbol, by its name: GCC numbers the symbol, ``hook.0``."""
	return merged(
		_unshared(group)
		for _, group in groupby(sorted(_numbered_symbols(symtab)), key=_static_name)
	)


def _numbered_symbols(symtab: SymbolTableSection | None) -> Iterator[_NumberedSymbol]:
	return flatmap(_numbered, filter(_local_object, _symbols(symtab)))


def _local_object(symbol: Symbol) -> bool:
	return symbol["st_info"]["type"] == "STT_OBJECT" and symbol["st_info"]["bind"] == "STB_LOCAL"


def _numbered(symbol: Symbol) -> tuple[_NumberedSymbol, ...]:
	named = _NUMBERED.match(symbol.name)
	return (_NumberedSymbol(name=named.group(1), symbol=symbol.name),) if named is not None else ()


def _unshared(group: Iterable[_NumberedSymbol]) -> dict[str, str]:
	numbered = tuple(group)
	return {numbered[0].name: numbered[0].symbol} if len(numbered) == 1 else {}


def _static_name(numbered: _NumberedSymbol) -> str:
	return numbered.name


def _typedef(die: DIE) -> dict[str, Record]:
	record = (
		_record(die) if die.tag == "DW_TAG_typedef" and "DW_AT_name" in die.attributes else None
	)
	return {die_name(die): record} if record is not None else {}


def _record(typedef: DIE) -> Record | None:
	"""The structure or union a typedef names, or None."""
	target = strip_qualifiers(die_type(typedef))
	if target is None or target.tag not in ("DW_TAG_structure_type", "DW_TAG_union_type"):
		return None
	key = layout_name(typedef)
	kind = "union" if target.tag == "DW_TAG_union_type" else "struct"
	return Record(name=key if key.startswith(("struct ", "union ")) else f"{kind} {key}", key=key)


def anonymous_name(key: str) -> str:
	"""A layout key as the plugin names its record: an anonymous one loses its DIE offset.

	>>> anonymous_name("union <anonymous>@0x1a2b"), anonymous_name("struct device")
	('union <anonymous>', 'struct device')
	"""
	return key.split("@", 1)[0] if f"{ANONYMOUS}@" in key else key
