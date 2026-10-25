# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""What the image's code adds to a build's call graph and frames."""

from collections import Counter
from typing import TYPE_CHECKING, assert_never

from salix import Struct

from dynamic_call_tree_resolution.call_sites import (
	calls_nothing,
	own_targets,
)
from dynamic_call_tree_resolution.callgraph import (
	CallEdge,
	EdgeKind,
)
from dynamic_call_tree_resolution.identity import (
	stack_name,
)
from dynamic_call_tree_resolution.linker import (
	Membership,
)
from dynamic_call_tree_resolution.loader import (
	defined_function_names,
)
from dynamic_call_tree_resolution.model import PrivilegedStack
from dynamic_call_tree_resolution.rtos.zephyr import NO_KCONFIG, Kconfig, kconfig, priority_levels
from dynamic_call_tree_resolution.stack_analysis import (
	INDIRECT_CALLEE,
	Frame,
	LevelSource,
	MeasuredFrame,
	OwnFrame,
	PriorityLevels,
	ThreadEntry,
	frame_key,
	own_thread_edges,
	thread_edges,
	thread_frames,
)
from dynamic_call_tree_resolution.stack_usage import StackUsage
from dynamic_call_tree_resolution.vsa.abi import normalized
from dynamic_call_tree_resolution.vsa.frames import (
	CodeMeasure,
	OwnCode,
	code_depths,
	own_frames,
)
from dynamic_call_tree_resolution.vsa.vectors import (
	HARD_FAULT,
)

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping
	from pathlib import Path

	from dynamic_call_tree_resolution.call_sites import CallerCandidates, ProgramResolution
	from dynamic_call_tree_resolution.cli.artifacts import Artifacts, Image
	from dynamic_call_tree_resolution.field_narrowing import NarrowedSpan
	from dynamic_call_tree_resolution.model import (
		Address,
		CallSite,
		RtosModel,
		SystemThread,
		ThreadRoot,
	)


class _CodeMeasures(Struct):
	"""Frames measured from code, and the calls that code adds to the graph."""

	frames: tuple[Frame, ...]
	edges: tuple[CallEdge, ...]


class _ThreadGraph(Struct):
	"""The calls and frames the RTOS model's threads add to the graph."""

	edges: tuple[CallEdge, ...]
	frames: tuple[Frame, ...]


def register_callers(
	image: Image, measure: CodeMeasure, sites: tuple[CallSite, ...], frames: tuple[StackUsage, ...]
) -> _CodeMeasures:
	framed = frozenset(frame_key(frame.function) for frame in frames)
	sites_by_start = Counter(
		normalized(site.caller_address, image.program.machine) for site in sites
	)
	owns = own_frames(
		measure,
		frozenset(
			start
			for start in sites_by_start
			if start in image.names and frame_key(image.names[start]) not in framed
		),
	)
	return _CodeMeasures(
		frames=tuple(
			OwnFrame(function=image.names[start], bytes=own.bytes) for start, own in owns.items()
		),
		edges=tuple(
			edge
			for start, own in sorted(owns.items())
			for edge in (
				*(
					CallEdge(caller=image.names[start], callee=INDIRECT_CALLEE)
					for _ in range(sites_by_start[start])
				),
				*_direct_edges(image, start, own),
			)
		),
	)


def unrecorded_sites(
	image: Image, sites: tuple[CallSite, ...], edges: tuple[CallEdge, ...]
) -> tuple[CallEdge, ...]:
	recorded = Counter(frame_key(edge.caller) for edge in edges if edge.callee == INDIRECT_CALLEE)
	return tuple(
		CallEdge(caller=caller, callee=INDIRECT_CALLEE)
		for caller, found in sorted(
			Counter(
				image.names[start]
				for site in sites
				if not calls_nothing(image.program, site)
				for start in (normalized(site.caller_address, image.program.machine),)
				if start in image.names
			).items()
		)
		for _ in range(found - recorded[frame_key(caller)])
	)


def _direct_edges(image: Image, start: Address, own: OwnCode) -> tuple[CallEdge, ...]:
	return tuple(
		CallEdge(caller=image.names[start], callee=image.names[callee], kind=EdgeKind.BINARY)
		for callee in sorted(own.callees)
		if callee in image.names
	)


def code_measures(
	image: Image,
	measure: CodeMeasure,
	edges: tuple[CallEdge, ...],
	frames: tuple[Frame, ...],
	roots: frozenset[str],
	attempted: frozenset[str],
) -> _CodeMeasures:
	unframed = (
		frozenset(frame_key(node) for edge in edges for node in (edge.caller, edge.callee))
		| frozenset(map(frame_key, roots))
	) - ({frame_key(frame.function) for frame in frames} | {INDIRECT_CALLEE} | attempted)
	if not unframed:
		return _CodeMeasures(frames=(), edges=())
	starts = {address: name for address, name in image.names.items() if frame_key(name) in unframed}
	depths = code_depths(measure, frozenset(starts))
	owns = own_frames(measure, frozenset(starts) - depths.keys())
	measured = tuple(
		address
		for key in unframed
		for addresses in (
			frozenset(address for address in starts if frame_key(starts[address]) == key),
		)
		if addresses and addresses <= depths.keys() | owns.keys()
		for address in addresses
	)
	found = _CodeMeasures(
		frames=tuple(
			MeasuredFrame(function=starts[address], bytes=depths[address])
			if address in depths
			else OwnFrame(function=starts[address], bytes=owns[address].bytes)
			for address in measured
		),
		edges=tuple(
			edge
			for address in sorted(measured)
			if address not in depths
			for edge in _direct_edges(image, address, owns[address])
		),
	)
	following = code_measures(
		image, measure, found.edges, (*frames, *found.frames), frozenset(), attempted | unframed
	)
	return _CodeMeasures(
		frames=(*found.frames, *following.frames), edges=(*found.edges, *following.edges)
	)


def image_function_names(artifacts: Artifacts, image: Image) -> frozenset[str]:
	match artifacts.membership:
		case Membership.LINKER:
			return artifacts.held
		case Membership.NAMES:
			return frozenset(
				map(frame_key, (*defined_function_names(image.elf), *image.names.values()))
			)
		case _ as unreachable:
			assert_never(unreachable)


def thread_graph(
	image: Image,
	model: RtosModel,
	edges: tuple[CallEdge, ...],
	frames: tuple[Frame, ...],
) -> _ThreadGraph:
	entries_by_thread = {
		**_thread_entries(image, model.threads, EdgeKind.THREAD),
		**_thread_entries(image, model.system_threads, EdgeKind.SYSTEM_THREAD),
	}
	match model.trampoline:
		case None:
			return _ThreadGraph(edges=(), frames=())
		case str() as trampoline:
			return _ThreadGraph(
				edges=thread_edges(edges, trampoline, entries_by_thread),
				frames=thread_frames(frames, trampoline, entries_by_thread),
			)
		case _ as unreachable:
			assert_never(unreachable)


class _StackEntry(Struct):
	"""A function a stack that is no thread's starts, and how it starts there."""

	function: Address
	kind: EdgeKind


def stack_nodes(image: Image, model: RtosModel) -> _ThreadGraph:
	entries = {
		**_privileged_entries(model.privileged_stack),
		**{
			stack.name: tuple(
				_StackEntry(function=entry, kind=EdgeKind.EXCEPTION) for entry in stack.entries
			)
			for stack in model.mode_stacks
		},
	}
	return _ThreadGraph(
		edges=tuple(
			CallEdge(
				caller=name,
				callee=frame_key(stack_name(image.program, image.names, entry.function)),
				kind=entry.kind,
			)
			for name, stack_entries in entries.items()
			for entry in stack_entries
		),
		frames=tuple(StackUsage(function=name, bytes=0, bounded=True) for name in entries),
	)


def _privileged_entries(stack: PrivilegedStack | None) -> Mapping[str, tuple[_StackEntry, ...]]:
	match stack:
		case None:
			return {}
		case PrivilegedStack():
			return {
				stack.name: (
					_StackEntry(function=stack.system_call, kind=EdgeKind.SYSTEM_CALL),
					_StackEntry(function=stack.user_start, kind=EdgeKind.USER_START),
				)
			}
		case _ as unreachable:
			assert_never(unreachable)


def _thread_entries(
	image: Image, threads: Iterable[ThreadRoot | SystemThread], started_by: EdgeKind
) -> Mapping[str, ThreadEntry]:
	return {
		thread.name: ThreadEntry(
			entry=frame_key(stack_name(image.program, image.names, thread.entry)),
			started_by=started_by,
		)
		for thread in threads
	}


class Targets(Struct):
	"""The whole image's targets for the indirect calls, by caller key."""

	candidates: CallerCandidates
	narrowed_by_caller: Mapping[str, Mapping[str, EdgeKind]]
	"""Reached only through a narrowing, with its edge."""
	narrowed_by_field: tuple[NarrowedSpan, ...]
	narrow_by_signature: bool


def own_thread_graphs(
	image: Image,
	model: RtosModel,
	resolution: ProgramResolution,
	edges: tuple[CallEdge, ...],
	targets: Targets,
) -> Mapping[str, tuple[CallEdge, ...]]:
	entries = {thread.name: thread.entry for thread in model.threads}
	match model.trampoline:
		case None:
			return {}
		case str() as trampoline:
			return {
				thread: own_thread_edges(
					edges,
					trampoline,
					thread,
					frame_key(stack_name(image.program, image.names, entries[thread])),
					own_targets(
						image.program,
						sites,
						resolution.assignments,
						image.names,
						targets.narrowed_by_field,
						narrow_by_signature=targets.narrow_by_signature,
						fallback=resolution.fallback,
					),
					targets.candidates.targets_by_caller,
					targets.candidates.fallback,
					targets.narrowed_by_caller,
				)
				for thread, sites in resolution.threads.items()
			}
		case _ as unreachable:
			assert_never(unreachable)


def build_kconfig(elf: Path) -> Kconfig:
	return (
		kconfig((elf.parent / ".config").read_text(errors="replace"))
		if (elf.parent / ".config").is_file()
		else NO_KCONFIG
	)


def exception_priority_levels(elf: Path, exceptions: Mapping[int, str]) -> PriorityLevels:
	match priority_levels(
		(elf.parent / "zephyr.dts").read_text(errors="replace")
		if (elf.parent / "zephyr.dts").is_file()
		else ""
	):
		case int() as count:
			return PriorityLevels(count=count, source=LevelSource.DEVICETREE)
		case None:
			return PriorityLevels(
				count=sum(number > HARD_FAULT for number in exceptions),
				source=LevelSource.VECTOR_TABLE,
			)
		case _ as unreachable:
			assert_never(unreachable)
