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

from dynamic_call_tree_resolution.callgraph import CallgraphFile, libcall_nodes
from dynamic_call_tree_resolution.loader import defined_function_names
from dynamic_call_tree_resolution.stack_analysis import frame_key
from dynamic_call_tree_resolution.stack_usage import StackUsageFile

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
_PROVIDED: Final = re.compile(
	r"^(?P<member>\S+\.a\([^)]+\))\s+\S+ \((?P<symbol>[^)]+)\)$", re.MULTILINE
)
_REMOVED_SECTION: Final = re.compile(
	r"removing unused section '(?P<section>\.text(?:\.[^']+)?)' in file '(?P<object>[^']+)'"
)
_KEPT_SECTION: Final = re.compile(
	r"^ (?P<section>\.text(?:\.\S+)?)\s+0x[0-9a-f]+\s+0x[0-9a-f]+ (?P<object>\S+)$",
	re.MULTILINE,
)
_DISCARDED_INPUT: Final = "\nDiscarded input sections"
_MEMORY_MAP: Final = "\nLinker script and memory map"


class LinkedObject(Struct, order=True):
	"""An object the final link took in: an archive member, or an object linked directly."""

	archive: str
	"""The archive's file name; empty for an object linked directly."""
	member: str
	"""The member's name, or the path of an object linked directly."""


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
	"""Each artifact's call edges and frames of the function copies the final link kept."""

	callgraphs: tuple[CallgraphFile, ...]
	usages: tuple[StackUsageFile, ...]
	dropped: Mapping[Linkage, frozenset[str]]
	"""The frame keys of the functions with no kept copy, by why the link has none."""
	phantom_libcalls: frozenset[str] = frozenset()
	"""Libcalls a kept function's ``.ci`` records but the final link did not keep."""


class LinkerRecords(Struct):
	"""The final link's object list and the function sections it garbage-collected."""

	linked: frozenset[LinkedObject]
	sections: Mapping[LinkedObject, Mapping[str, bool]]
	"""Each linked object's code sections, and whether the link kept each one."""
	providers: Mapping[str, LinkedObject] = {}
	"""The archive member the link included for each symbol that pulled one in."""
	defined: frozenset[str] = frozenset()
	"""The function symbols the image defines."""


def linker_records(elf: Path) -> LinkerRecords | None:
	link_map = elf.parent / MAP
	listing = elf.parent / GC_LISTING
	if not link_map.is_file() or not listing.is_file():
		return None
	text = link_map.read_text()
	return LinkerRecords(
		providers=_providers(text.split(_DISCARDED_INPUT, 1)[0]),
		defined=defined_function_names(elf),
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
			linked: {entry.section: entry.kept for entry in group}
			for linked, group in groupby(
				sorted(
					(
						*(
							_linked_section(match, kept=False)
							for match in _REMOVED_SECTION.finditer(listing.read_text())
						),
						*(
							_linked_section(match, kept=True)
							for match in _KEPT_SECTION.finditer(text.partition(_MEMORY_MAP)[2])
						),
					)
				),
				key=_object_of_section,
			)
		},
	)


def _providers(included: str) -> Mapping[str, LinkedObject]:
	"""The member each symbol pulled in, whether the map wraps its reference line or not.

	>>> _providers(
	...     chr(10).join(
	...         (
	...             "/sdk/libgcc.a(_aeabi_uldivmod.o)",
	...             "                              zephyr/libzephyr.a(clock.c.obj) (__aeabi_uldivmod)",
	...             "zephyr/libk.a(a.c.obj)        app/libapp.a(main.c.obj) (k_a)",
	...         )
	...     )
	... )
	{'__aeabi_uldivmod': LinkedObject(archive='libgcc.a', member='_aeabi_uldivmod.o'), 'k_a': LinkedObject(archive='libk.a', member='a.c.obj')}
	"""
	return {
		match["symbol"]: _linked_object(match["member"]) for match in _PROVIDED.finditer(included)
	}


def libcall_linkage(records: LinkerRecords, name: str) -> Linkage:
	"""Whether the final link kept the code of a libcall a ``.ci`` records as ``<built-in>``."""
	match records.providers.get(name):
		case _ if name in records.defined:
			return Linkage.IN_IMAGE
		case None:
			return Linkage.NEVER_LINKED
		case member:
			return _section_linkage(tuple(records.sections.get(member, {}).values()))


def _matched_object(match: re.Match[str]) -> LinkedObject:
	return _linked_object(match["object"])


class _LinkedSection(Struct, order=True):
	"""A code section of a linked object, and whether the final link kept it."""

	linked: LinkedObject
	section: str
	kept: bool


def _linked_section(match: re.Match[str], *, kept: bool) -> _LinkedSection:
	return _LinkedSection(
		linked=_linked_object(match["object"]), section=match["section"], kept=kept
	)


def _object_of_section(entry: _LinkedSection) -> LinkedObject:
	return entry.linked


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
			LinkedObject(archive=f"lib{target}.a", member=f"{posixpath.basename(stem)}.obj"),
			_linked_object(f"{directory.lstrip('/')}/CMakeFiles/{target}.dir/{stem}.obj"),
		)
		if separator
		else ()
	)


def _linked_object(name: str) -> LinkedObject:
	match _ARCHIVE_MEMBER.fullmatch(posixpath.normpath(name)):
		case None:
			return LinkedObject(archive="", member=posixpath.normpath(name).lstrip("/"))
		case member:
			return LinkedObject(archive=member["archive"], member=member["member"])


class _LinkedEdge(Struct):
	"""A ``.ci`` edge, with how the final link kept its caller."""

	edge: CallEdge
	linkage: Linkage
	phantom: bool
	"""Whether the callee is a libcall the final link did not keep."""


class _LinkedEdges(Struct):
	"""One ``.ci`` file's edges, each with how the final link kept its caller."""

	path: Path
	edges: tuple[_LinkedEdge, ...]


class _LinkedUsage(Struct):
	"""A ``.su`` record, with how the final link kept its function."""

	usage: StackUsage
	linkage: Linkage


class _LinkedUsages(Struct):
	"""One ``.su`` file's records, each with how the final link kept its function."""

	path: Path
	usages: tuple[_LinkedUsage, ...]


def held_artifacts(
	records: LinkerRecords,
	build_directory: Path,
	callgraphs: tuple[CallgraphFile, ...],
	usages: tuple[StackUsageFile, ...],
) -> HeldArtifacts:
	edges = tuple(
		_LinkedEdges(
			path=file.path,
			edges=tuple(
				_LinkedEdge(
					edge=edge,
					linkage=linkage(
						records, build_directory, file.path, edge.caller.rsplit(":", 1)[-1]
					),
					phantom=edge.callee in not_kept,
				)
				for edge in file.edges
			),
		)
		for file in callgraphs
		for not_kept in (
			frozenset(
				name
				for name in libcall_nodes(file.path)
				if libcall_linkage(records, name) is not Linkage.IN_IMAGE
			),
		)
	)
	frames = tuple(
		_LinkedUsages(
			path=file.path,
			usages=tuple(
				_LinkedUsage(
					usage=usage,
					linkage=linkage(records, build_directory, file.path, usage.function),
				)
				for usage in file.usages
			),
		)
		for file in usages
	)
	held = frozenset(
		(
			*(
				frame_key(linked.edge.caller)
				for linked_file in edges
				for linked in linked_file.edges
				if linked.linkage is Linkage.IN_IMAGE
			),
			*(
				frame_key(linked.usage.function)
				for linked_file in frames
				for linked in linked_file.usages
				if linked.linkage is Linkage.IN_IMAGE
			),
		)
	)
	discarded = (
		frozenset(
			frame_key(linked.usage.function)
			for linked_file in frames
			for linked in linked_file.usages
			if linked.linkage is Linkage.DISCARDED
		)
		- held
	)
	return HeldArtifacts(
		callgraphs=tuple(
			CallgraphFile(
				path=linked_file.path,
				edges=tuple(
					linked.edge
					for linked in linked_file.edges
					if linked.linkage is Linkage.IN_IMAGE and not linked.phantom
				),
			)
			for linked_file in edges
		),
		usages=tuple(
			StackUsageFile(
				path=linked_file.path,
				usages=tuple(
					linked.usage
					for linked in linked_file.usages
					if linked.linkage is Linkage.IN_IMAGE
				),
			)
			for linked_file in frames
		),
		dropped={
			Linkage.DISCARDED: discarded,
			Linkage.NEVER_LINKED: frozenset(
				frame_key(linked.usage.function)
				for linked_file in frames
				for linked in linked_file.usages
				if linked.linkage is Linkage.NEVER_LINKED
			)
			- held
			- discarded,
		},
		phantom_libcalls=frozenset(
			linked.edge.callee
			for linked_file in edges
			for linked in linked_file.edges
			if linked.linkage is Linkage.IN_IMAGE and linked.phantom
		),
	)
