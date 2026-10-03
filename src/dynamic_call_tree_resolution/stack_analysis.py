# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Worst-case stack depth over a resolved call graph."""

from __future__ import annotations

import re
from collections import Counter
from enum import StrEnum
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Final, assert_never

from salix import Struct

from dynamic_call_tree_resolution.callgraph import CallEdge, EdgeKind
from dynamic_call_tree_resolution.stack_usage import StackUsage

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Mapping


class Bounded(Struct):
	"""A depth that bounds every path of the call graph as given."""

	bytes: int


class Unbounded(Struct):
	"""A depth that bounds only from below."""

	at_least: int
	recursion: frozenset[str]
	"""Reachable functions on a cycle."""
	unmeasured: frozenset[str]
	"""Reachable functions without a ``.su`` record."""
	dynamic: frozenset[str]
	"""Reachable frames GCC could not bound."""
	unresolved: frozenset[str]
	"""Reachable callers of an indirect call without candidates."""


class StackReport(Struct):
	"""One worst-case stack depth."""

	entry: str
	bound: Bounded | Unbounded


class Reason(StrEnum):
	"""What keeps a depth from bounding every path."""

	RECURSION = "recursion"
	UNMEASURED = "unmeasured"
	DYNAMIC = "dynamic"
	UNRESOLVED = "unresolved"


type _Reasons = frozenset[tuple[Reason, str]]


class StackGraph(Struct):
	"""A call graph with the frames and depths its stack bounds come from."""

	adjacency: Mapping[str, frozenset[str]]
	frame_by_name: Mapping[str, StackUsage]
	components: Mapping[str, frozenset[str]]
	"""Each node's strongly connected component."""
	depths: Mapping[str, int]
	reasons: Mapping[str, _Reasons]
	kinds: Mapping[tuple[str, str], frozenset[EdgeKind]]
	roots: tuple[str, ...]


class PathStep(Struct):
	"""One function on a deepest path."""

	function: str
	frame: int
	cumulative: int
	edge: frozenset[EdgeKind]
	"""How the previous step calls this one."""
	flags: frozenset[Reason]
	"""What this function's own frame, calls and cycle add to an unbounded depth."""


class _Forward(Struct):
	"""A large component's depths along the preorder from where a path entered it."""

	depths: Mapping[str, int]
	position: Mapping[str, int]


_MAX_CYCLE_SIZE: Final = 16

INDIRECT_CALLEE: Final = "__indirect_call"


def expand_indirect_calls(
	edges: Iterable[CallEdge],
	targets_by_caller: Mapping[str, frozenset[str]],
	fallback: frozenset[str],
	*,
	exact: bool = False,
	field_targets_by_caller: Mapping[str, frozenset[str]] | None = None,
) -> tuple[CallEdge, ...]:
	edges_tuple = tuple(edges)
	graph_targets = partial(_graph_names, _raw_names_by_key(edges_tuple))
	return tuple(
		CallEdge(caller=edge.caller, callee=callee, kind=kind)
		for edge in edges_tuple
		for callee, kind in (
			_expansion(
				targets_by_caller.get(frame_key(edge.caller), frozenset[str]()),
				(field_targets_by_caller or {}).get(frame_key(edge.caller), frozenset[str]()),
				fallback,
				graph_targets,
				exact=exact,
			)
			or [(INDIRECT_CALLEE, edge.kind)]
			if edge.callee == INDIRECT_CALLEE
			else ((edge.callee, edge.kind),)
		)
	)


def _expansion(
	targets: frozenset[str],
	field_targets: frozenset[str],
	fallback: frozenset[str],
	graph_targets: Callable[[str], frozenset[str]],
	*,
	exact: bool,
) -> list[tuple[str, EdgeKind]]:
	candidates = targets - {INDIRECT_CALLEE}
	return sorted(
		(graph_target, _indirect_kind(target, candidates, field_targets))
		for target in candidates
		| (fallback if INDIRECT_CALLEE in targets or not exact else frozenset[str]())
		for graph_target in graph_targets(target)
	)


def _indirect_kind(
	target: str, candidates: frozenset[str], field_targets: frozenset[str]
) -> EdgeKind:
	return (
		EdgeKind.FIELD
		if target in field_targets
		else EdgeKind.CANDIDATE
		if target in candidates
		else EdgeKind.FALLBACK
	)


def thread_edges(
	edges: Iterable[CallEdge], trampoline: str, entries_by_thread: Mapping[str, str]
) -> tuple[CallEdge, ...]:
	edges_tuple = tuple(edges)
	names_by_key = _raw_names_by_key(edges_tuple)
	return tuple(
		thread_edge
		for thread, entry in entries_by_thread.items()
		for thread_edge in (
			*(
				CallEdge(caller=thread, callee=edge.callee, kind=edge.kind)
				for edge in edges_tuple
				if frame_key(edge.caller) == trampoline and edge.callee != INDIRECT_CALLEE
			),
			*(
				CallEdge(caller=thread, callee=name, kind=EdgeKind.THREAD)
				for name in sorted(_graph_names(names_by_key, entry))
			),
		)
	)


class ThreadTargets(Struct):
	"""What one thread's own analysis resolved, by caller key."""

	reached: frozenset[str]
	targets_by_caller: Mapping[str, frozenset[str]]
	sites_by_caller: Mapping[str, int]
	field_targets_by_caller: Mapping[str, frozenset[str]] = {}
	"""The targets each caller reaches only through ``--narrow-by-field``."""


def own_thread_edges(
	edges: Iterable[CallEdge],
	trampoline: str,
	thread: str,
	entry: str,
	own: ThreadTargets,
	targets_by_caller: Mapping[str, frozenset[str]],
	fallback: frozenset[str],
	field_targets_by_caller: Mapping[str, frozenset[str]] | None = None,
) -> tuple[CallEdge, ...]:
	edges_tuple = tuple(edges)
	graph_targets = partial(_graph_names, _raw_names_by_key(edges_tuple))
	node = partial(_thread_node, thread, own.reached)
	complete = _complete_callers(edges_tuple, own)
	return (
		*(
			CallEdge(caller=thread, callee=node(edge.callee), kind=edge.kind)
			for edge in edges_tuple
			if frame_key(edge.caller) == trampoline and edge.callee != INDIRECT_CALLEE
		),
		*(
			CallEdge(caller=thread, callee=node(name), kind=EdgeKind.THREAD)
			for name in sorted(graph_targets(entry))
		),
		*(
			CallEdge(caller=_in_thread(thread, edge.caller), callee=callee, kind=kind)
			for edge in edges_tuple
			if frame_key(edge.caller) in own.reached
			for callee, kind in (
				(
					sorted(
						(
							node(graph_target),
							_indirect_kind(
								target,
								own.targets_by_caller[frame_key(edge.caller)],
								own.field_targets_by_caller.get(
									frame_key(edge.caller), frozenset[str]()
								),
							),
						)
						for target in own.targets_by_caller[frame_key(edge.caller)]
						for graph_target in graph_targets(target)
					)
					if frame_key(edge.caller) in complete
					else _expansion(
						targets_by_caller.get(frame_key(edge.caller), frozenset[str]()),
						(field_targets_by_caller or {}).get(
							frame_key(edge.caller), frozenset[str]()
						),
						fallback,
						graph_targets,
						exact=False,
					)
					or [(INDIRECT_CALLEE, edge.kind)]
				)
				if edge.callee == INDIRECT_CALLEE
				else ((node(edge.callee), edge.kind),)
			)
		),
	)


def _complete_callers(edges: tuple[CallEdge, ...], own: ThreadTargets) -> frozenset[str]:
	indirect_edges = Counter(
		frame_key(edge.caller) for edge in edges if edge.callee == INDIRECT_CALLEE
	)
	return frozenset(
		caller
		for caller, targets in own.targets_by_caller.items()
		if INDIRECT_CALLEE not in targets
		and own.sites_by_caller.get(caller, 0) >= indirect_edges[caller]
	)


def _thread_node(thread: str, reached: frozenset[str], name: str) -> str:
	return _in_thread(thread, name) if frame_key(name) in reached else name


def _in_thread(thread: str, name: str) -> str:
	"""Name a function's node in one thread's own tree; its frame stays the function's.

	>>> frame_key(_in_thread("thermal_tid", "/zephyr/drivers/i2c/i2c_emul.c:i2c_emul_transfer"))
	'i2c_emul_transfer'
	>>> frame_key(_in_thread("thermal_tid", "i2c_write_read.constprop.0"))
	'i2c_write_read'
	"""
	return f"{thread}/:{name}"


def thread_frames(
	frames: Iterable[StackUsage], trampoline: str, threads: Iterable[str]
) -> tuple[StackUsage, ...]:
	frames_tuple = tuple(frames)
	return tuple(
		StackUsage(function=thread, bytes=usage.bytes, bounded=usage.bounded)
		for thread in threads
		for usage in frames_tuple
		if frame_key(usage.function) == trampoline
	)


def _raw_names_by_key(edges: tuple[CallEdge, ...]) -> Mapping[str, frozenset[str]]:
	return {
		key: frozenset(name for _, name in group)
		for key, group in groupby(
			sorted(
				{(frame_key(name), name) for edge in edges for name in (edge.caller, edge.callee)}
			),
			key=_first,
		)
	}


def _first(pair: tuple[str, str]) -> str:
	return pair[0]


def _graph_names(names_by_key: Mapping[str, frozenset[str]], target: str) -> frozenset[str]:
	return names_by_key.get(frame_key(target), frozenset({target}))


def worst_case_depths(
	edges: Iterable[CallEdge],
	frames: Iterable[StackUsage],
	*,
	entry_edges: Iterable[CallEdge] | None = None,
) -> tuple[StackReport, ...]:
	return stack_reports(stack_graph(edges, frames, entry_edges=entry_edges))


def stack_graph(
	edges: Iterable[CallEdge],
	frames: Iterable[StackUsage],
	*,
	entry_edges: Iterable[CallEdge] | None = None,
) -> StackGraph:
	edges_tuple = tuple(edges)
	frames_tuple = tuple(frames)
	frame_by_name = _frames_by_bare_name(frames_tuple)
	adjacency = _adjacency(edges_tuple)
	root_adjacency = _adjacency(entry_edges) if entry_edges is not None else adjacency
	components = _strongly_connected_components(adjacency)
	root_callees = _callees(root_adjacency)
	root_graph_nodes = {frame_key(node) for node in set(root_adjacency) | root_callees}
	return StackGraph(
		adjacency=adjacency,
		frame_by_name=frame_by_name,
		components={node: frozenset(component) for component in components for node in component},
		depths=_depths_by_component(adjacency, frame_by_name, components),
		reasons=_reasons_by_component(adjacency, frame_by_name, components),
		kinds={
			pair: frozenset(edge.kind for edge in group)
			for pair, group in groupby(sorted(edges_tuple, key=_pair), key=_pair)
		},
		roots=tuple(
			sorted(
				(root_adjacency.keys() - root_callees)
				| ({frame_key(usage.function) for usage in frames_tuple} - root_graph_nodes)
			)
		),
	)


def stack_reports(graph: StackGraph) -> tuple[StackReport, ...]:
	return tuple(sorted((_report(graph, root) for root in graph.roots), key=_report_order))


def thread_reports(
	reports: Iterable[StackReport], own_graphs: Mapping[str, StackGraph]
) -> tuple[StackReport, ...]:
	return tuple(
		sorted(
			(
				_report(own_graphs[report.entry], report.entry)
				if report.entry in own_graphs
				else report
				for report in reports
			),
			key=_report_order,
		)
	)


def deepest_path(graph: StackGraph, entry: str) -> tuple[PathStep, ...]:
	root = min(
		(node for node in graph.roots if frame_key(node) == entry),
		key=partial(_root_rank, graph),
		default=None,
	)
	if root is None:
		raise ValueError(f"{entry} is not an entry of the call graph")
	return _walk(graph, root, frozenset(), _entered(graph, root), 0, frozenset(), {})


def _root_depth(graph: StackGraph, root: str) -> int:
	return graph.depths[root] if root in graph.depths else graph.frame_by_name[root].bytes


def _root_reasons(graph: StackGraph, root: str) -> _Reasons:
	return (
		graph.reasons[root]
		if root in graph.reasons
		else _own_reasons(root, graph.adjacency, graph.frame_by_name)
	)


def _report(graph: StackGraph, root: str) -> StackReport:
	return StackReport(
		entry=frame_key(root), bound=_bound(_root_depth(graph, root), _root_reasons(graph, root))
	)


def _root_rank(graph: StackGraph, root: str) -> tuple[bool, int, str]:
	return (*_report_order(_report(graph, root)), root)


def _pair(edge: CallEdge) -> tuple[str, str]:
	return (edge.caller, edge.callee)


def _walk(
	graph: StackGraph,
	node: str,
	path: frozenset[str],
	forward: _Forward | None,
	cumulative: int,
	edge: frozenset[EdgeKind],
	within: dict[tuple[str, frozenset[str]], int],
) -> tuple[PathStep, ...]:
	frame = _frame_size(graph, node)
	step = PathStep(
		function=frame_key(node),
		frame=frame,
		cumulative=cumulative + frame,
		edge=edge,
		flags=_flags(graph, node),
	)
	callee = max(
		_callee_depths(graph, node, path, forward, within), key=_callee_depth, default=None
	)
	if callee is None:
		return (step,)
	enters = callee[0] not in _members(graph, node)
	return (
		step,
		*_walk(
			graph,
			callee[0],
			frozenset() if enters else path | {node},
			_entered(graph, callee[0]) if enters else forward,
			cumulative + frame,
			graph.kinds[(node, callee[0])],
			within,
		),
	)


def _callee_depths(
	graph: StackGraph,
	node: str,
	path: frozenset[str],
	forward: _Forward | None,
	within: dict[tuple[str, frozenset[str]], int],
) -> tuple[tuple[str, int], ...]:
	members = _members(graph, node)
	callees = sorted(graph.adjacency.get(node, frozenset()) - {INDIRECT_CALLEE})
	if forward is not None:
		return tuple(
			(callee, graph.depths[callee] if callee not in members else forward.depths[callee])
			for callee in callees
			if callee not in members or forward.position[callee] > forward.position[node]
		)
	return tuple(
		(
			callee,
			graph.depths[callee]
			if callee not in members
			else _depth(
				callee,
				graph.adjacency,
				graph.frame_by_name,
				path | {node},
				graph.depths,
				members,
				within,
			),
		)
		for callee in callees
		if callee not in path | {node}
	)


def _callee_depth(callee: tuple[str, int]) -> int:
	return callee[1]


def _entered(graph: StackGraph, node: str) -> _Forward | None:
	members = _members(graph, node)
	if len(members) <= _MAX_CYCLE_SIZE:
		return None
	order = _preorder(node, members, graph.adjacency)
	return _Forward(
		depths=_forward_depths(order, members, graph.adjacency, graph.frame_by_name, graph.depths),
		position={member: index for index, member in enumerate(order)},
	)


def _members(graph: StackGraph, node: str) -> frozenset[str]:
	return graph.components.get(node, frozenset({node}))


def _frame_size(graph: StackGraph, node: str) -> int:
	frame = graph.frame_by_name.get(frame_key(node))
	return frame.bytes if frame is not None else 0


def _flags(graph: StackGraph, node: str) -> frozenset[Reason]:
	return frozenset(
		reason for reason, _ in _own_reasons(node, graph.adjacency, graph.frame_by_name)
	) | (
		frozenset({Reason.RECURSION})
		if len(_members(graph, node)) > 1 or node in graph.adjacency.get(node, frozenset())
		else frozenset[Reason]()
	)


def _adjacency(edges: Iterable[CallEdge]) -> dict[str, frozenset[str]]:
	callees_by_name: dict[str, set[str]] = {}
	for edge in edges:
		callees_by_name.setdefault(edge.caller, set()).add(edge.callee)
	return {caller: frozenset(callees) for caller, callees in callees_by_name.items()}


def _bound(depth: int, reasons: _Reasons) -> Bounded | Unbounded:
	return (
		Unbounded(
			at_least=depth,
			recursion=_names(reasons, Reason.RECURSION),
			unmeasured=_names(reasons, Reason.UNMEASURED),
			dynamic=_names(reasons, Reason.DYNAMIC),
			unresolved=_names(reasons, Reason.UNRESOLVED),
		)
		if reasons
		else Bounded(bytes=depth)
	)


def _names(reasons: _Reasons, reason: Reason) -> frozenset[str]:
	return frozenset(name for kind, name in reasons if kind is reason)


def _report_order(report: StackReport) -> tuple[bool, int]:
	match report.bound:
		case Unbounded(at_least=at_least):
			return (False, -at_least)
		case Bounded(bytes=depth):
			return (True, -depth)
		case _ as unreachable:
			assert_never(unreachable)


def _own_reasons(
	node: str,
	adjacency: Mapping[str, frozenset[str]],
	frame_by_name: Mapping[str, StackUsage],
) -> _Reasons:
	frame = frame_by_name.get(frame_key(node))
	return frozenset(
		(reason, frame_key(node))
		for reason, applies in (
			(Reason.UNMEASURED, frame is None and node != INDIRECT_CALLEE),
			(Reason.DYNAMIC, frame is not None and not frame.bounded),
			(Reason.UNRESOLVED, INDIRECT_CALLEE in adjacency.get(node, frozenset())),
		)
		if applies
	)


def _reasons_by_component(
	adjacency: Mapping[str, frozenset[str]],
	frame_by_name: Mapping[str, StackUsage],
	components: tuple[tuple[str, ...], ...],
) -> Mapping[str, _Reasons]:
	closure: dict[str, _Reasons] = {}
	for component in components:
		members = frozenset(component)
		reasons = frozenset[tuple[Reason, str]]().union(
			*(_own_reasons(member, adjacency, frame_by_name) for member in component),
			*(
				closure[callee]
				for member in component
				for callee in adjacency.get(member, frozenset())
				if callee not in members
			),
			(
				frozenset((Reason.RECURSION, frame_key(member)) for member in component)
				if len(component) > 1
				or any(member in adjacency.get(member, frozenset()) for member in component)
				else frozenset()
			),
		)
		closure.update(dict.fromkeys(component, reasons))
	return closure


def _depths_by_component(
	adjacency: Mapping[str, frozenset[str]],
	frame_by_name: Mapping[str, StackUsage],
	components: tuple[tuple[str, ...], ...],
) -> Mapping[str, int]:
	memo: dict[str, int] = {}
	known = {node for component in components for node in component}
	all_nodes = set(adjacency) | _callees(adjacency)
	for component in (*components, *((node,) for node in sorted(all_nodes - known))):
		in_component = frozenset(component)
		within: dict[tuple[str, frozenset[str]], int] = {}
		memo.update(
			_rooted_depths(component, adjacency, frame_by_name, memo)
			if len(component) > _MAX_CYCLE_SIZE
			else {
				node: _depth(
					node, adjacency, frame_by_name, frozenset(), memo, in_component, within
				)
				for node in component
			}
		)
	return memo


def _rooted_depths(
	component: tuple[str, ...],
	adjacency: Mapping[str, frozenset[str]],
	frame_by_name: Mapping[str, StackUsage],
	memo: Mapping[str, int],
) -> dict[str, int]:
	members = frozenset(component)
	return {
		root: _forward_depths(
			_preorder(root, members, adjacency), members, adjacency, frame_by_name, memo
		)[root]
		for root in component
	}


def _preorder(
	root: str, members: frozenset[str], adjacency: Mapping[str, frozenset[str]]
) -> tuple[str, ...]:
	order: list[str] = []
	seen: set[str] = set()
	stack = [root]
	while stack:
		node = stack.pop()
		if node in seen:
			continue
		seen.add(node)
		order.append(node)
		stack.extend(sorted((adjacency.get(node, frozenset()) & members) - seen, reverse=True))
	return tuple(order)


def _forward_depths(
	order: tuple[str, ...],
	members: frozenset[str],
	adjacency: Mapping[str, frozenset[str]],
	frame_by_name: Mapping[str, StackUsage],
	memo: Mapping[str, int],
) -> dict[str, int]:
	position = {node: index for index, node in enumerate(order)}
	depths: dict[str, int] = {}
	for node in reversed(order):
		frame = frame_by_name.get(frame_key(node))
		depths[node] = (frame.bytes if frame is not None else 0) + max(
			(
				memo[callee] if callee not in members else depths[callee]
				for callee in adjacency.get(node, frozenset())
				if callee not in members or position[callee] > position[node]
			),
			default=0,
		)
	return depths


def _depth(
	function: str,
	adjacency: Mapping[str, frozenset[str]],
	frame_by_name: Mapping[str, StackUsage],
	path: frozenset[str],
	memo: Mapping[str, int],
	in_component: frozenset[str],
	within: dict[tuple[str, frozenset[str]], int],
) -> int:
	key = (function, path & in_component)
	if key in within:
		return within[key]
	frame = frame_by_name.get(frame_key(function))
	current_path = path | {function}
	children: list[int] = []
	for callee in sorted(adjacency.get(function, frozenset())):
		if callee in current_path:
			continue
		cached = None if callee in in_component else memo.get(callee)
		children.append(
			cached
			if cached is not None
			else _depth(callee, adjacency, frame_by_name, current_path, memo, in_component, within)
		)
	depth = (frame.bytes if frame is not None else 0) + max(children, default=0)
	within[key] = depth
	return depth


def _frame_bytes(usage: StackUsage) -> int:
	return usage.bytes


def _frame_name(usage: StackUsage) -> str:
	return frame_key(usage.function)


def _callees(adjacency: Mapping[str, frozenset[str]]) -> frozenset[str]:
	return frozenset(callee for callees in adjacency.values() for callee in callees)


def _strongly_connected_components(
	adjacency: Mapping[str, frozenset[str]],
) -> tuple[tuple[str, ...], ...]:
	indices: dict[str, int] = {}
	lowlinks: dict[str, int] = {}
	stack: list[str] = []
	on_stack: set[str] = set()
	components: list[tuple[str, ...]] = []

	def strong_connect(node: str) -> None:
		index = len(indices)
		indices[node] = index
		lowlinks[node] = index
		stack.append(node)
		on_stack.add(node)
		for neighbor in sorted(adjacency.get(node, frozenset())):
			if neighbor not in indices:
				strong_connect(neighbor)
				lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
			elif neighbor in on_stack:
				lowlinks[node] = min(lowlinks[node], indices[neighbor])
		if lowlinks[node] == indices[node]:
			component: list[str] = []
			while True:
				member = stack.pop()
				on_stack.remove(member)
				component.append(member)
				if member == node:
					break
			components.append(tuple(component))

	for node in sorted(adjacency):
		if node not in indices:
			strong_connect(node)
	return tuple(components)


def frame_key(name: str) -> str:
	"""Reduce a VCG or ``.su`` symbol to its bare assembly form.

	>>> frame_key("/abs/path/file.c:func")
	'func'
	>>> frame_key("func.isra.0")
	'func'
	>>> frame_key("func.constprop.0.isra.0"), frame_key("func.constprop.isra")
	('func', 'func')
	>>> frame_key("func.localalias"), frame_key("func.cold"), frame_key("func.lto_priv.0")
	('func', 'func', 'func')
	>>> frame_key("func.part_of"), frame_key("func.isra.0.x")
	('func.part_of', 'func.isra.0.x')
	"""
	return re.sub(
		r"(?:\.(?:isra|constprop|part|localalias|cold|lto_priv)(?:\.\d+)?)+$",
		"",
		name.rsplit(":", 1)[-1] if "/" in name else name,
	)


def _frames_by_bare_name(frames: Iterable[StackUsage]) -> Mapping[str, StackUsage]:
	return {key: _merge_frame_records(records) for key, records in _grouped_frames(frames).items()}


def _grouped_frames(frames: Iterable[StackUsage]) -> dict[str, list[StackUsage]]:
	return {
		key: list(group) for key, group in groupby(sorted(frames, key=_frame_name), key=_frame_name)
	}


def _merge_frame_records(records: list[StackUsage]) -> StackUsage:
	largest = max(records, key=_frame_bytes)
	return StackUsage(
		function=largest.function,
		bytes=largest.bytes,
		bounded=all(usage.bounded for usage in records),
	)
