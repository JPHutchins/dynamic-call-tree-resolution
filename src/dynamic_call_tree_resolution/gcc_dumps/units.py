# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""A build's C units, each with the alias dump GCC wrote beside its object and its object's types."""

from __future__ import annotations

import shlex
from glob import escape
from pathlib import Path
from typing import TYPE_CHECKING, Final

import msgspec
from elftools.elf.elffile import ELFFile
from salix import Struct

from dynamic_call_tree_resolution.loader.dwarf import die_name, iter_dies
from dynamic_call_tree_resolution.loader.layouts import (
	die_type,
	layout_name,
	strip_qualifiers,
	structure_layouts,
)
from dynamic_call_tree_resolution.model import ANONYMOUS

if TYPE_CHECKING:
	from collections.abc import Mapping

	from elftools.dwarf.die import DIE

	from dynamic_call_tree_resolution.model import StructureLayout

DUMP_FLAG: Final = "-fdump-tree-alias-lineno"


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


class DumpedUnit(Struct):
	"""One C unit of the build, with its alias dump when GCC wrote one."""

	unit: str
	"""The build's path of the unit's object, without its suffix."""
	source: Path
	object: Path
	dump: Path | None
	"""None for a unit without a function, for which GCC writes no dump."""


class _CompileCommand(msgspec.Struct):
	"""One entry of a CMake compile_commands.json."""

	directory: str
	command: str
	file: str
	output: str


def dumped_units(build_directory: Path) -> tuple[DumpedUnit, ...]:
	"""The build's C units, each paired with its alias dump.

	Raises:
		ValueError: when a unit was compiled without the dump flag, or its dump is older than its
			source, so that it was written before the source last changed.
	"""
	commands = tuple(
		command
		for command in msgspec.json.decode(
			(build_directory / "compile_commands.json").read_bytes(),
			type=tuple[_CompileCommand, ...],
		)
		if command.file.endswith(".c")
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
	output = (Path(command.directory) / command.output).relative_to(command.directory)
	dumps = sorted(
		(build_directory / output.parent).glob(
			f"{escape(output.stem)}{escape(Path(command.file).suffix)}.*.alias"
		)
	)
	return DumpedUnit(
		unit=output.with_suffix("").as_posix(),
		source=Path(command.directory) / command.file,
		object=build_directory / output,
		dump=dumps[-1] if dumps else None,
	)


def unit_types(object_path: Path) -> UnitTypes:
	"""What one unit's object says about the types its dump names.

	Raises:
		ValueError: when the object has no debug information to name its types.
	"""
	with object_path.open("rb") as stream:
		elf = ELFFile(stream)
		if not elf.has_dwarf_info():
			raise ValueError(f"{object_path} has no debug information: build with -g")
		dwarf = elf.get_dwarf_info(relocate_dwarf_sections=False)
		dies = tuple(
			die
			for compilation_unit in dwarf.iter_CUs()
			for die in iter_dies(compilation_unit.get_top_DIE())
		)
		return UnitTypes(
			layouts=structure_layouts(dwarf),
			typedefs={
				die_name(die): record
				for die in dies
				if die.tag == "DW_TAG_typedef" and "DW_AT_name" in die.attributes
				for record in (_record(die),)
				if record is not None
			},
			variables={
				die_name(die): layout_name(die_type(die))
				for die in reversed(
					tuple(
						child
						for compilation_unit in dwarf.iter_CUs()
						for child in compilation_unit.get_top_DIE().iter_children()
					)
				)
				if die.tag == "DW_TAG_variable" and "DW_AT_name" in die.attributes
			},
			functions=frozenset(
				die_name(die)
				for die in dies
				if die.tag == "DW_TAG_subprogram" and "DW_AT_name" in die.attributes
			),
		)


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
