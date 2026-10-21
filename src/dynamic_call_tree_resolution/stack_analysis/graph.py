# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The stack graph of a call graph and its frames."""

from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING, Final, assert_never

from dynamic_call_tree_resolution.callgraph import CallEdge, EdgeKind, call_pair
from dynamic_call_tree_resolution.stack_analysis.depths import (
	all_callees,
	depths_by_component,
	strongly_connected_components,
)
from dynamic_call_tree_resolution.stack_analysis.values import (
	INDIRECT_CALLEE,
	AssumedFrame,
	Frame,
	FunctionReason,
	MeasuredFrame,
	OwnFrame,
	Reason,
	Reasons,
	StackGraph,
	frame_key,
)
from dynamic_call_tree_resolution.stack_usage import StackUsage

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping


def stack_graph(
	edges: Iterable[CallEdge],
	frames: Iterable[Frame],
	*,
	entry_edges: Iterable[CallEdge] | None = None,
	assumed_no_recursion: frozenset[str] = frozenset(),
	assumed_unwritten: frozenset[str] = frozenset(),
	hardware_roots: frozenset[str] = frozenset(),
) -> StackGraph:
	given = tuple(edges)
	edges_tuple = tuple(edge for edge in given if not _assumed_away(edge, assumed_no_recursion))
	assumed = frozenset(
		frame_key(edge.caller) for edge in given if _assumed_away(edge, assumed_no_recursion)
	)
	frames_tuple = tuple(frames)
	frame_by_name = _frames_by_bare_name(frames_tuple)
	adjacency = _adjacency(edges_tuple)
	root_adjacency = (
		_adjacency(edge for edge in entry_edges if not _assumed_away(edge, assumed_no_recursion))
		if entry_edges is not None
		else adjacency
	)
	components = strongly_connected_components(adjacency)
	root_callees = all_callees(root_adjacency)
	root_graph_nodes = {frame_key(node) for node in set(root_adjacency) | root_callees}
	narrowings = _narrowings(edges_tuple, assumed_unwritten)
	return StackGraph(
		adjacency=adjacency,
		frame_by_name=frame_by_name,
		components={node: frozenset(component) for component in components for node in component},
		depths=depths_by_component(adjacency, frame_by_name, components),
		reasons=_reasons_by_component(adjacency, frame_by_name, components, assumed, narrowings),
		assumed_no_recursion=assumed,
		narrowings=narrowings,
		kinds={
			pair: frozenset(edge.kind for edge in group)
			for pair, group in groupby(sorted(edges_tuple, key=call_pair), key=call_pair)
		},
		roots=tuple(
			sorted(
				(root_adjacency.keys() - root_callees)
				| (
					{
						frame_key(record.function)
						for frame in frames_tuple
						for record in (_record(frame),)
						if record is not None
					}
					- root_graph_nodes
				)
				| frozenset(map(frame_key, hardware_roots))
			)
		),
	)


_NARROWED: Final = {
	EdgeKind.FIELD: Reason.NARROWED_BY_FIELD,
	EdgeKind.SIGNATURE: Reason.NARROWED_BY_SIGNATURE,
}


def _narrowings(
	edges: tuple[CallEdge, ...], assumed_unwritten: frozenset[str]
) -> Mapping[str, frozenset[Reason]]:
	narrowed = {
		caller: reasons
		for caller, group in groupby(sorted(edges, key=call_pair), key=_edge_caller)
		for reasons in (
			frozenset(_NARROWED[edge.kind] for edge in group if edge.kind in _NARROWED),
		)
		if reasons
	}
	return (
		{
			**narrowed,
			**{
				node: narrowed.get(node, frozenset[Reason]()) | {Reason.ASSUMED_UNWRITTEN}
				for edge in edges
				for node in (edge.caller, edge.callee)
				if frame_key(node) in assumed_unwritten
			},
		}
		if assumed_unwritten
		else narrowed
	)


def _edge_caller(edge: CallEdge) -> str:
	return edge.caller


def _adjacency(edges: Iterable[CallEdge]) -> dict[str, frozenset[str]]:
	callees_by_name: dict[str, set[str]] = {}
	for edge in edges:
		callees_by_name.setdefault(edge.caller, set()).add(edge.callee)
	return {caller: frozenset(callees) for caller, callees in callees_by_name.items()}


def _assumed_away(edge: CallEdge, assumed_no_recursion: frozenset[str]) -> bool:
	return (
		frame_key(edge.caller) == frame_key(edge.callee)
		and frame_key(edge.callee) in assumed_no_recursion
	)


def _reasons_by_component(
	adjacency: Mapping[str, frozenset[str]],
	frame_by_name: Mapping[str, Frame],
	components: tuple[tuple[str, ...], ...],
	assumed_no_recursion: frozenset[str],
	narrowings: Mapping[str, frozenset[Reason]],
) -> Mapping[str, Reasons]:
	closure: dict[str, Reasons] = {}
	for component in components:
		members = frozenset(component)
		reasons = frozenset[FunctionReason]().union(
			*(
				own_reasons(member, adjacency, frame_by_name, assumed_no_recursion, narrowings)
				for member in component
			),
			*(
				closure[callee]
				for member in component
				for callee in adjacency.get(member, frozenset())
				if callee not in members
			),
			(
				frozenset(
					FunctionReason(reason=Reason.RECURSION, function=frame_key(member))
					for member in component
				)
				if len(component) > 1
				or any(member in adjacency.get(member, frozenset()) for member in component)
				else frozenset()
			),
		)
		closure.update(dict.fromkeys(component, reasons))
	return closure


def own_reasons(
	node: str,
	adjacency: Mapping[str, frozenset[str]],
	frame_by_name: Mapping[str, Frame],
	assumed_no_recursion: frozenset[str],
	narrowings: Mapping[str, frozenset[Reason]],
) -> Reasons:
	return frozenset(
		FunctionReason(reason=reason, function=frame_key(node))
		for reason in (
			*_frame_reasons(frame_by_name.get(frame_key(node)), node),
			*narrowings.get(node, frozenset[Reason]()),
			*(
				(Reason.ASSUMED_NO_RECURSION,)
				if frame_key(node) in assumed_no_recursion
				else tuple[Reason, ...]()
			),
			*(
				(Reason.UNRESOLVED,)
				if INDIRECT_CALLEE in adjacency.get(node, frozenset())
				else tuple[Reason, ...]()
			),
		)
	)


def _frame_reasons(frame: Frame | None, node: str) -> tuple[Reason, ...]:
	match frame:
		case None:
			return tuple[Reason, ...]() if node == INDIRECT_CALLEE else (Reason.UNMEASURED,)
		case StackUsage(bounded=bounded):
			return tuple[Reason, ...]() if bounded else (Reason.DYNAMIC,)
		case MeasuredFrame() | OwnFrame():
			return (Reason.MEASURED,)
		case AssumedFrame():
			return (Reason.ASSUMED_FRAME,)
		case _ as unreachable:
			assert_never(unreachable)


def _frames_by_bare_name(frames: Iterable[Frame]) -> Mapping[str, Frame]:
	return {key: _merge_frame_records(records) for key, records in _grouped_frames(frames).items()}


def _grouped_frames(frames: Iterable[Frame]) -> dict[str, list[Frame]]:
	return {
		key: list(group) for key, group in groupby(sorted(frames, key=_frame_name), key=_frame_name)
	}


def _merge_frame_records(records: list[Frame]) -> Frame:
	match tuple(record for frame in records if (record := _record(frame)) is not None):
		case ():
			return max(records, key=_frame_bytes)
		case usages:
			largest = max(usages, key=_frame_bytes)
			return StackUsage(
				function=largest.function,
				bytes=largest.bytes,
				bounded=all(usage.bounded for usage in usages),
			)


def _frame_bytes(frame: Frame) -> int:
	return frame.bytes


def _frame_name(frame: Frame) -> str:
	return frame_key(frame.function)


def _record(frame: Frame) -> StackUsage | None:
	match frame:
		case StackUsage():
			return frame
		case MeasuredFrame() | OwnFrame() | AssumedFrame():
			return None
		case _ as unreachable:
			assert_never(unreachable)
