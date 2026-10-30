# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""A build's call graph and frames, and the image they come from."""

from typing import TYPE_CHECKING, Final

from salix import Struct

from dynamic_call_tree_resolution.callgraph import (
	CallEdge,
	callgraph_files,
	callgraph_locations,
)
from dynamic_call_tree_resolution.identity import (
	Identities,
	canonical_edges,
	canonical_frames,
	identities,
	stack_names,
)
from dynamic_call_tree_resolution.linker import (
	Linkage,
	Membership,
	held_artifacts,
	linker_records,
)
from dynamic_call_tree_resolution.loader import (
	load,
)
from dynamic_call_tree_resolution.stack_analysis import (
	Frame,
	PriorityLevels,
	frame_key,
)
from dynamic_call_tree_resolution.stack_usage import (
	StackUsage,
	stack_usage_files,
	stack_usage_locations,
)

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path

	from dynamic_call_tree_resolution.call_sites import ProgramResolution
	from dynamic_call_tree_resolution.callgraph import CallgraphFile
	from dynamic_call_tree_resolution.model import Address, Program, RtosModel, UnresolvedSlot
	from dynamic_call_tree_resolution.report import SlotCounts
	from dynamic_call_tree_resolution.stack_usage import StackUsageFile


class HandlerNames(Struct):
	"""The stack-graph names of the handlers an A- or R-profile core's handler stack runs."""

	interrupts: Mapping[int, str]
	"""Each connected interrupt's handler, by interrupt number."""
	spurious: Mapping[int, str]
	"""An unconnected line's fatal handler, by interrupt number."""
	wrapper_calls: tuple[str, ...]


class Expansion(Struct):
	"""A build's call graph and frames, its indirect calls expanded against its ELF image."""

	expanded: tuple[CallEdge, ...]
	in_image_edges: tuple[CallEdge, ...]
	frames: tuple[Frame, ...]
	threads: frozenset[str]
	hardware_roots: frozenset[str]
	"""The handlers only the vector table holds, by their names in the stack graph."""
	exceptions: Mapping[int, str]
	"""Each exception's handler on an interrupt stack the RTOS model names, by exception
	number."""
	priority_levels: PriorityLevels
	own_edges: Mapping[str, tuple[CallEdge, ...]]
	"""Each thread's tree from its own analysis, by thread."""
	indirect_sites: int
	program: Program
	rtos: RtosModel
	resolution: ProgramResolution
	counts: SlotCounts
	unresolved: tuple[UnresolvedSlot, ...]
	image_functions: frozenset[str]
	membership: Membership
	dropped: Mapping[Linkage, frozenset[str]]
	"""The entries the linker left out of the image, by why; empty under name membership."""
	phantom_libcalls: frozenset[str]
	"""Libcalls a kept function's ``.ci`` records but the final link did not keep."""
	assumed_unwritten: AssumedUnwritten
	assumed_upward_frame_writes: AssumedCallers
	handlers: HandlerNames | None = None


class AssumedCallers(Struct):
	"""The callers whose indirect calls the assumptions changed, by frame key."""

	callers: frozenset[str]
	thread_callers: Mapping[str, frozenset[str]]
	"""The same, in each thread's own analysis."""


NO_ASSUMED_CALLERS: Final = AssumedCallers(callers=frozenset(), thread_callers={})


class AssumedUnwritten(Struct):
	"""What ``--assume-unwritten`` changed."""

	objects: tuple[str, ...]
	contradicted: tuple[str, ...]
	"""The objects a tracked store writes after all."""
	changed: AssumedCallers


class Artifacts(Struct):
	"""The ``.ci`` edges and ``.su`` frames a stack graph is built from."""

	edges: tuple[CallEdge, ...]
	frames: tuple[StackUsage, ...]
	membership: Membership
	held: frozenset[str]
	"""The frame keys of the functions the artifacts hold; the linker's choice under its
	membership."""
	dropped: Mapping[Linkage, frozenset[str]]
	phantom_libcalls: frozenset[str]
	"""Libcalls a kept function's ``.ci`` records but the final link did not keep."""


class Image(Struct):
	"""An ELF image, its analysis model and its functions' names in the stack graph."""

	elf: Path
	program: Program
	names: Mapping[Address, str]


def optional_image(elf: Path | None) -> Image | None:
	return loaded(elf) if elf is not None else None


def loaded(elf: Path) -> Image:
	program = load(elf)
	return Image(elf=elf, program=program, names=stack_names(program))


def build_artifacts(build_directory: Path, image: Image | None) -> Artifacts:
	known = identities(image.program, image.names) if image is not None else None
	match linker_records(image.elf) if image is not None else None:
		case None:
			return _named_artifacts(
				known,
				callgraph_files(build_directory),
				stack_usage_files(build_directory),
				Membership.NAMES,
				{},
				frozenset(),
			)
		case records:
			held = held_artifacts(
				records,
				build_directory,
				callgraph_files(build_directory),
				stack_usage_files(build_directory),
			)
			return _named_artifacts(
				known,
				held.callgraphs,
				held.usages,
				Membership.LINKER,
				held.dropped,
				held.phantom_libcalls,
			)


def _named_artifacts(
	known: Identities | None,
	callgraphs: tuple[CallgraphFile, ...],
	usages: tuple[StackUsageFile, ...],
	membership: Membership,
	dropped: Mapping[Linkage, frozenset[str]],
	phantom_libcalls: frozenset[str],
) -> Artifacts:
	edges = tuple(
		edge for file in callgraphs for edge in _named_edges(known, file.path, file.edges)
	)
	frames = tuple(
		frame for file in usages for frame in _named_frames(known, file.path, file.usages)
	)
	return Artifacts(
		edges=edges,
		frames=frames,
		membership=membership,
		held=frozenset(
			(
				*(frame_key(edge.caller) for edge in edges),
				*(frame_key(frame.function) for frame in frames),
			)
		),
		dropped=dropped,
		phantom_libcalls=phantom_libcalls,
	)


def _named_edges(
	known: Identities | None, path: Path, edges: tuple[CallEdge, ...]
) -> tuple[CallEdge, ...]:
	return (
		canonical_edges(known, path.name.removesuffix(".ci"), callgraph_locations(path), edges)
		if known is not None
		else edges
	)


def _named_frames(
	known: Identities | None, path: Path, frames: tuple[StackUsage, ...]
) -> tuple[StackUsage, ...]:
	return (
		canonical_frames(known, path.name.removesuffix(".su"), stack_usage_locations(path), frames)
		if known is not None
		else frames
	)
