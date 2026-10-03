# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Which objects and function sections the final link kept, from its map and gc listing."""

from __future__ import annotations

import posixpath
import re
from enum import StrEnum
from itertools import groupby
from typing import TYPE_CHECKING, Final

from salix import Struct

from dynamic_call_tree_resolution.stack_analysis import frame_key

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path

	from dynamic_call_tree_resolution.callgraph import CallEdge
	from dynamic_call_tree_resolution.stack_usage import StackUsage

MAP: Final = "zephyr_final.map"
GC_LISTING: Final = "gc-sections.txt"

_ARCHIVE_MEMBER: Final = re.compile(r"(?:.*/)?(?P<archive>lib[^/]+\.a)\((?P<member>.+)\)")
_INCLUDED_MEMBER: Final = re.compile(r"^(?P<object>\S+\.a\(\S+\.obj\))", re.MULTILINE)
_LOADED_OBJECT: Final = re.compile(r"^LOAD (?P<object>\S+\.obj)$", re.MULTILINE)
_REMOVED_SECTION: Final = re.compile(
	r"removing unused section '(?P<section>\.text\.[^']+)' in file '(?P<object>[^']+)'"
)
_KEPT_SECTION: Final = re.compile(
	r"^ (?P<section>\.text\.\S+)\s+0x[0-9a-f]+\s+0x[0-9a-f]+ (?P<object>\S+)$", re.MULTILINE
)
_DISCARDED_INPUT: Final = "\nDiscarded input sections"
_MEMORY_MAP: Final = "\nLinker script and memory map"

type LinkedObject = tuple[str, str]
"""An archive member as (archive file name, member), or an object linked directly as ("", path)."""


class Linkage(StrEnum):
	"""Whether the final link kept a function."""

	IN_IMAGE = "in image"
	DISCARDED = "discarded"
	NEVER_LINKED = "never linked"


class Membership(StrEnum):
	"""What decides whether the image holds a function."""

	LINKER = "linker"
	NAMES = "names"


class HeldArtifacts(Struct):
	"""The call edges and frames of the function copies the final link kept."""

	edges: tuple[CallEdge, ...]
	frames: tuple[StackUsage, ...]
	held: frozenset[str]
	"""The frame keys of every function with a kept copy."""
	dropped: Mapping[Linkage, frozenset[str]]
	"""The frame keys of the functions with no kept copy, by why the link has none."""


class LinkerRecords(Struct):
	"""The final link's object list and the function sections it garbage-collected."""

	linked: frozenset[LinkedObject]
	sections: Mapping[LinkedObject, Mapping[str, bool]]
	"""Each linked object's function sections, and whether the link kept each one."""


def linker_records(elf: Path) -> LinkerRecords | None:
	link_map = elf.parent / MAP
	listing = elf.parent / GC_LISTING
	if not link_map.is_file() or not listing.is_file():
		return None
	text = link_map.read_text()
	return LinkerRecords(
		linked=frozenset(
			map(
				_matched_object,
				(
					*_INCLUDED_MEMBER.finditer(text.split(_DISCARDED_INPUT, 1)[0]),
					*_LOADED_OBJECT.finditer(text),
				),
			)
		),
		sections={
			linked: {section: kept for _, section, kept in group}
			for linked, group in groupby(
				sorted(
					(
						*(
							(*_section(match), False)
							for match in _REMOVED_SECTION.finditer(listing.read_text())
						),
						*(
							(*_section(match), True)
							for match in _KEPT_SECTION.finditer(text.partition(_MEMORY_MAP)[2])
						),
					)
				),
				key=_object_of_section,
			)
		},
	)


def _matched_object(match: re.Match[str]) -> LinkedObject:
	return _linked_object(match["object"])


def _section(match: re.Match[str]) -> tuple[LinkedObject, str]:
	return (_linked_object(match["object"]), match["section"])


def _object_of_section(entry: tuple[LinkedObject, str, bool]) -> LinkedObject:
	return entry[0]


def linkage(
	records: LinkerRecords, build_directory: Path, artifact: Path, function: str
) -> Linkage:
	match _artifact_objects(build_directory, artifact):
		case ():
			return Linkage.IN_IMAGE
		case candidates:
			match next((linked for linked in candidates if linked in records.linked), None):
				case None:
					return Linkage.NEVER_LINKED
				case linked:
					return _section_linkage(
						tuple(
							kept
							for section, kept in records.sections.get(linked, {}).items()
							if _names(section, function)
						)
					)


def _section_linkage(kept: tuple[bool, ...]) -> Linkage:
	return Linkage.DISCARDED if kept and not any(kept) else Linkage.IN_IMAGE


def _names(section: str, function: str) -> bool:
	"""Whether a function section holds the function a ``.su`` or ``.ci`` record names.

	``.su`` drops a clone's number, which the section keeps.

	>>> _names(".text.inplace_realloc.isra.0", "inplace_realloc.isra")
	True
	>>> _names(".text.inplace_realloc.isra.0", "inplace_realloc.isra.0")
	True
	>>> _names(".text.inplace_realloc", "inplace")
	False
	"""
	name = section.removeprefix(".text.")
	return name == function or (
		name.startswith(f"{function}.") and name.removeprefix(f"{function}.").isdigit()
	)


def _artifact_objects(build_directory: Path, artifact: Path) -> tuple[LinkedObject, ...]:
	directory, separator, target_and_source = (
		f"/{artifact.relative_to(build_directory).as_posix()}".partition("/CMakeFiles/")
	)
	target, _, source = target_and_source.partition(".dir/")
	stem = source.rsplit(".", 1)[0]
	return (
		(
			(f"lib{target}.a", f"{posixpath.basename(stem)}.obj"),
			_linked_object(f"{directory.lstrip('/')}/CMakeFiles/{target}.dir/{stem}.obj"),
		)
		if separator
		else ()
	)


def _linked_object(name: str) -> LinkedObject:
	match _ARCHIVE_MEMBER.fullmatch(posixpath.normpath(name)):
		case None:
			return ("", posixpath.normpath(name).lstrip("/"))
		case member:
			return (member["archive"], member["member"])


def held_artifacts(
	records: LinkerRecords,
	build_directory: Path,
	callgraphs: tuple[tuple[Path, tuple[CallEdge, ...]], ...],
	usages: tuple[tuple[Path, tuple[StackUsage, ...]], ...],
) -> HeldArtifacts:
	edges = tuple(
		(edge, linkage(records, build_directory, path, edge.caller.rsplit(":", 1)[-1]))
		for path, file_edges in callgraphs
		for edge in file_edges
	)
	frames = tuple(
		(usage, linkage(records, build_directory, path, usage.function))
		for path, file_usages in usages
		for usage in file_usages
	)
	held = frozenset(
		(
			*(frame_key(edge.caller) for edge, linked in edges if linked is Linkage.IN_IMAGE),
			*(frame_key(usage.function) for usage, linked in frames if linked is Linkage.IN_IMAGE),
		)
	)
	discarded = (
		frozenset(
			frame_key(usage.function) for usage, linked in frames if linked is Linkage.DISCARDED
		)
		- held
	)
	return HeldArtifacts(
		edges=tuple(edge for edge, linked in edges if linked is Linkage.IN_IMAGE),
		frames=tuple(usage for usage, linked in frames if linked is Linkage.IN_IMAGE),
		held=held,
		dropped={
			Linkage.DISCARDED: discarded,
			Linkage.NEVER_LINKED: frozenset(
				frame_key(usage.function)
				for usage, linked in frames
				if linked is Linkage.NEVER_LINKED
			)
			- held
			- discarded,
		},
	)
