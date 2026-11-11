# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""What the image's code adds to a build's call graph and frames."""

from collections import Counter
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Final, assert_never

from salix import Struct

from dynamic_call_tree_resolution.call_sites import (
	calls_nothing,
	own_targets,
)
from dynamic_call_tree_resolution.callgraph import (
	CallEdge,
	EdgeKind,
	edge_nodes,
)
from dynamic_call_tree_resolution.composition import flatmap
from dynamic_call_tree_resolution.identity import (
	stack_name,
)
from dynamic_call_tree_resolution.linker import (
	Membership,
)
from dynamic_call_tree_resolution.loader import (
	defined_function_names,
)
from dynamic_call_tree_resolution.model import HandlerStack, InstructionFamily, PrivilegedStack
from dynamic_call_tree_resolution.rtos.zephyr import NO_KCONFIG, Kconfig, kconfig, priority_levels
from dynamic_call_tree_resolution.stack_analysis import (
	INDIRECT_CALLEE,
	Frame,
	LevelSource,
	MeasuredFrame,
	OwnFrame,
	PriorityLevels,
	ThreadEntry,
	complete_callers,
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
	from collections.abc import Iterable, Iterator, Mapping
	from collections.abc import Set as AbstractSet
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
		edges=tuple(flatmap(partial(_start_edges, image, sites_by_start, owns), sorted(owns))),
	)


def _start_edges(
	image: Image,
	sites_by_start: Mapping[Address, int],
	owns: Mapping[Address, OwnCode],
	start: Address,
) -> Iterator[CallEdge]:
	return chain(
		_indirect_edges(image.names[start], sites_by_start[start]),
		_direct_edges(image, start, owns[start]),
	)


def _indirect_edges(caller: str, count: int) -> Iterator[CallEdge]:
	return (CallEdge(caller=caller, callee=INDIRECT_CALLEE) for _ in range(count))


def unrecorded_sites(
	image: Image, sites: tuple[CallSite, ...], edges: tuple[CallEdge, ...]
) -> tuple[CallEdge, ...]:
	recorded = Counter(frame_key(edge.caller) for edge in edges if edge.callee == INDIRECT_CALLEE)
	starts = (
		normalized(site.caller_address, image.program.machine)
		for site in sites
		if not calls_nothing(image.program, site)
	)
	found = Counter(image.names[start] for start in starts if start in image.names)
	return tuple(
		flatmap(
			lambda caller: _indirect_edges(caller, found[caller] - recorded[frame_key(caller)]),
			sorted(found),
		)
	)


def resolved_callers(
	image: Image,
	sites: tuple[CallSite, ...],
	targets_by_caller: Mapping[str, frozenset[str]],
	edges: tuple[CallEdge, ...],
) -> frozenset[str]:
	match image.program.machine.family:
		case InstructionFamily.ARM:
			return complete_callers(
				edges,
				targets_by_caller,
				Counter(
					frame_key(stack_name(image.program, image.names, site.caller_address))
					for site in sites
				),
			)
		case InstructionFamily.X86:
			return frozenset()
		case _ as unreachable:
			assert_never(unreachable)


def _direct_edges(image: Image, start: Address, own: OwnCode) -> Iterator[CallEdge]:
	return (
		CallEdge(caller=image.names[start], callee=image.names[callee], kind=EdgeKind.BINARY)
		for callee in sorted(own.callees)
		if callee in image.names
	)


def _complete_addresses(
	starts: Mapping[Address, str], measurable: AbstractSet[Address], key: str
) -> frozenset[Address]:
	addresses = frozenset(address for address in starts if frame_key(starts[address]) == key)
	return addresses if addresses and addresses <= measurable else frozenset()


def code_measures(
	image: Image,
	measure: CodeMeasure,
	edges: tuple[CallEdge, ...],
	frames: tuple[Frame, ...],
	roots: frozenset[str],
	attempted: frozenset[str],
) -> _CodeMeasures:
	unframed = (
		frozenset(map(frame_key, flatmap(edge_nodes, edges))) | frozenset(map(frame_key, roots))
	) - ({frame_key(frame.function) for frame in frames} | {INDIRECT_CALLEE} | attempted)
	if not unframed:
		return _CodeMeasures(frames=(), edges=())
	starts = {address: name for address, name in image.names.items() if frame_key(name) in unframed}
	depths = code_depths(measure, frozenset(starts))
	owns = own_frames(measure, frozenset(starts) - depths.keys())
	measurable = depths.keys() | owns.keys()
	measured = tuple(flatmap(lambda key: _complete_addresses(starts, measurable, key), unframed))
	found = _CodeMeasures(
		frames=tuple(
			MeasuredFrame(function=starts[address], bytes=depths[address])
			if address in depths
			else OwnFrame(function=starts[address], bytes=owns[address].bytes)
			for address in measured
		),
		edges=tuple(
			flatmap(
				lambda address: _direct_edges(image, address, owns[address]),
				(address for address in sorted(measured) if address not in depths),
			)
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


class _StackNode(Struct):
	"""What starts on a stack that is no thread's, or on a path onto one, and its own frame."""

	entries: tuple[_StackEntry, ...]
	frame: int = 0


def stack_nodes(image: Image, model: RtosModel) -> _ThreadGraph:
	nodes = {
		**_privileged_node(model.privileged_stack),
		**{
			stack.name: _StackNode(
				entries=tuple(
					_StackEntry(function=entry, kind=EdgeKind.EXCEPTION) for entry in stack.entries
				)
			)
			for stack in model.mode_stacks
		},
		**_system_call_node(model.handler_stack),
	}
	return _ThreadGraph(
		edges=tuple(flatmap(lambda name: _node_edges(image, name, nodes[name]), nodes)),
		frames=tuple(
			StackUsage(function=name, bytes=node.frame, bounded=True)
			for name, node in nodes.items()
		),
	)


def _node_edges(image: Image, name: str, node: _StackNode) -> Iterator[CallEdge]:
	return (
		CallEdge(
			caller=name,
			callee=frame_key(stack_name(image.program, image.names, entry.function)),
			kind=entry.kind,
		)
		for entry in node.entries
	)


def _privileged_node(stack: PrivilegedStack | None) -> Mapping[str, _StackNode]:
	match stack:
		case None:
			return {}
		case PrivilegedStack():
			return {
				stack.name: _StackNode(
					entries=(
						_StackEntry(function=stack.system_call, kind=EdgeKind.SYSTEM_CALL),
						_StackEntry(function=stack.user_start, kind=EdgeKind.USER_START),
					)
				)
			}
		case _ as unreachable:
			assert_never(unreachable)


SYSTEM_CALL_NODE: Final = "SVC"


def _system_call_node(stack: HandlerStack | None) -> Mapping[str, _StackNode]:
	match stack:
		case None:
			return {}
		case HandlerStack():
			return {
				SYSTEM_CALL_NODE: _StackNode(
					entries=tuple(
						_StackEntry(function=function, kind=EdgeKind.EXCEPTION)
						for function in stack.system_calls
					),
					frame=stack.system_call_frame,
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
	complete: frozenset[str]
	"""The callers whose indirect calls drop the fallback across the whole image."""


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
					targets.complete,
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
