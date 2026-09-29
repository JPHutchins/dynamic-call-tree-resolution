# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Worst-case stack depth over a resolved call graph."""

from __future__ import annotations

import re
from enum import Enum, auto
from itertools import groupby
from typing import TYPE_CHECKING, Final, assert_never

from salix import Struct

from dynamic_call_tree_resolution.callgraph import CallEdge
from dynamic_call_tree_resolution.stack_usage import StackUsage

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping


class Bounded(Struct):
	"""A depth that bounds every path of the call graph as given."""

	bytes: int


class Unbounded(Struct):
	"""A depth that bounds only from below, with the functions that break the bound.

	``recursion`` holds the reachable functions on a cycle, ``unmeasured``
	the reachable functions without a ``.su`` record, ``dynamic`` the
	reachable frames GCC could not bound, and ``unresolved`` the reachable
	callers of an indirect call without candidates.
	"""

	at_least: int
	recursion: frozenset[str]
	unmeasured: frozenset[str]
	dynamic: frozenset[str]
	unresolved: frozenset[str]


class StackReport(Struct):
	"""Worst-case stack depth of one entry point."""

	entry: str
	bound: Bounded | Unbounded


class _Reason(Enum):
	RECURSION = auto()
	UNMEASURED = auto()
	DYNAMIC = auto()
	UNRESOLVED = auto()


type _Reasons = frozenset[tuple[_Reason, str]]


_MAX_CYCLE_SIZE: Final = 16

INDIRECT_CALLEE: Final = "__indirect_call"


def expand_indirect_calls(
	edges: Iterable[CallEdge],
	targets_by_caller: Mapping[str, frozenset[str]],
	fallback: frozenset[str],
	*,
	exact: bool = False,
) -> tuple[CallEdge, ...]:
	"""Replace GCC ``__indirect_call`` placeholders with resolved targets.

	Each caller's placeholders expand to the union of the candidates of the
	call sites extracted from that caller's code and ``fallback`` (every
	address-taken function): per-caller candidates refine on top of the
	fallback, never replace it, so sites the extractor missed cannot vanish
	from the bound. With ``exact`` the fallback is not added on top, so
	a caller's placeholders expand to its per-caller candidates alone;
	those still hold the fallback for its unresolved sites. A placeholder
	left with no targets
	stays, so its caller's entries report it as unresolved.

	With ``exact``, candidate names are also translated to every raw ``.ci``
	graph name with the same bare key — static functions are path-qualified
	there — so an expanded edge lands on the node whose own outgoing edges
	connect the deeper graph. (In the sound mode the bare names stay: the
	translation connects the fallback union into real recursive cycles
	through cbprintf-style output pointers, which the sound bound must
	flag rather than silently drop.)
	"""
	edges_tuple = tuple(edges)
	raw_names_by_key: dict[str, set[str]] = {}
	for edge in edges_tuple:
		for name in (edge.caller, edge.callee):
			raw_names_by_key.setdefault(frame_key(name), set()).add(name)

	def graph_targets(target: str) -> frozenset[str]:
		if not exact:
			return frozenset({target})
		raw_names = raw_names_by_key.get(frame_key(target))
		return frozenset(raw_names) if raw_names is not None else frozenset({target})

	return tuple(
		CallEdge(caller=edge.caller, callee=callee)
		for edge in edges_tuple
		for callee in (
			sorted(
				graph_target
				for target in sorted(
					targets_by_caller.get(frame_key(edge.caller), frozenset[str]())
					| (frozenset[str]() if exact else fallback)
				)
				for graph_target in graph_targets(target)
			)
			or [INDIRECT_CALLEE]
			if edge.callee == INDIRECT_CALLEE
			else (edge.callee,)
		)
	)


def worst_case_depths(
	edges: Iterable[CallEdge],
	frames: Iterable[StackUsage],
	*,
	entry_edges: Iterable[CallEdge] | None = None,
) -> tuple[StackReport, ...]:
	"""Compute worst-case stack depths for every entry point.

	Entry points are functions with no incoming call edges, plus
	``.su``-recorded functions with no call edges at all (leaf callbacks
	and ISRs). When ``entry_edges`` (the unexpanded graph) is given, the
	root test uses it while the depths still use the expanded ``edges``:
	expansion adds only sound fallback edges, so a thread function whose
	only incoming edges are fallback unions keeps its per-entry report.
	The depth is the deepest path with each cycle broken at its back edge
	and each frame without a ``.su`` record counted as 0 bytes; it is a
	bound only when nothing the entry reaches is recursive, unmeasured,
	unboundedly dynamic, or an unresolved indirect call. Unbounded entries
	come first, then bounded ones, each deepest first.
	"""
	frames_tuple = tuple(frames)
	frame_by_name = _frames_by_bare_name(frames_tuple)
	adjacency = _adjacency(edges)
	root_adjacency = _adjacency(entry_edges) if entry_edges is not None else adjacency
	components = _strongly_connected_components(adjacency)
	depths = _depths_by_component(adjacency, frame_by_name, components)
	reasons = _reasons_by_component(adjacency, frame_by_name, components)
	root_callees = _callees(root_adjacency)
	root_graph_nodes = {frame_key(node) for node in set(root_adjacency) | root_callees}
	roots = sorted(
		(root_adjacency.keys() - root_callees)
		| ({frame_key(usage.function) for usage in frames_tuple} - root_graph_nodes)
	)
	return tuple(
		sorted(
			(
				StackReport(
					entry=frame_key(entry),
					bound=_bound(
						depths[entry] if entry in depths else frame_by_name[entry].bytes,
						reasons[entry]
						if entry in reasons
						else _own_reasons(entry, adjacency, frame_by_name),
					),
				)
				for entry in roots
			),
			key=_report_order,
		)
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
			recursion=_names(reasons, _Reason.RECURSION),
			unmeasured=_names(reasons, _Reason.UNMEASURED),
			dynamic=_names(reasons, _Reason.DYNAMIC),
			unresolved=_names(reasons, _Reason.UNRESOLVED),
		)
		if reasons
		else Bounded(bytes=depth)
	)


def _names(reasons: _Reasons, reason: _Reason) -> frozenset[str]:
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
	"""What breaks the bound at ``node`` itself: its frame and its indirect calls."""
	frame = frame_by_name.get(frame_key(node))
	return frozenset(
		(reason, frame_key(node))
		for reason, applies in (
			(_Reason.UNMEASURED, frame is None and node != INDIRECT_CALLEE),
			(_Reason.DYNAMIC, frame is not None and not frame.bounded),
			(_Reason.UNRESOLVED, INDIRECT_CALLEE in adjacency.get(node, frozenset())),
		)
		if applies
	)


def _reasons_by_component(
	adjacency: Mapping[str, frozenset[str]],
	frame_by_name: Mapping[str, StackUsage],
	components: tuple[tuple[str, ...], ...],
) -> Mapping[str, _Reasons]:
	"""Every node's reasons closed over all it reaches, one SCC at a time, sinks first.

	Members of one component reach each other, so they share one set; a
	component of several functions, or one calling itself, is recursion.
	"""
	closure: dict[str, _Reasons] = {}
	for component in components:
		members = frozenset(component)
		reasons = frozenset[tuple[_Reason, str]]().union(
			*(_own_reasons(member, adjacency, frame_by_name) for member in component),
			*(
				closure[callee]
				for member in component
				for callee in adjacency.get(member, frozenset())
				if callee not in members
			),
			(
				frozenset((_Reason.RECURSION, frame_key(member)) for member in component)
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
	"""Per-node depths, computed one SCC at a time, sinks first.

	Tarjan emits strongly connected components in reverse topological
	order of the condensation, so every callee outside a node's own
	component is already computed when the node is reached. Only paths
	within one component need the path-based recursion, which keeps the
	search polynomial on diamond-heavy call graphs. A component larger
	than ``_MAX_CYCLE_SIZE`` is recursion either way and its longest
	simple path is exponential to search, so its members take the longest
	paths of an acyclic part of it instead.
	"""
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
	"""Each member's longest path over its own depth-first walk's forward edges.

	Edges toward nodes a walk from the member discovers later hold the
	walk's tree and no cycle, so every path over them is a real simple path.
	"""
	members = frozenset(component)
	return {
		root: _forward_depth(
			_preorder(root, members, adjacency), members, adjacency, frame_by_name, memo
		)
		for root in component
	}


def _preorder(
	root: str, members: frozenset[str], adjacency: Mapping[str, frozenset[str]]
) -> tuple[str, ...]:
	"""A depth-first preorder of ``members`` from ``root``, callees in sorted order."""
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


def _forward_depth(
	order: tuple[str, ...],
	members: frozenset[str],
	adjacency: Mapping[str, frozenset[str]],
	frame_by_name: Mapping[str, StackUsage],
	memo: Mapping[str, int],
) -> int:
	"""The longest path from ``order[0]`` over edges toward later nodes of ``order``."""
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
	return depths[order[0]]


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
	"""SCCs in reverse topological order of the condensation (sinks first).

	Nodes and neighbors are visited in sorted order, so each component's
	member order is the same in every process.
	"""
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
	"""Reduce a VCG or ``.su`` function name to its bare assembly name.

	GCC's ``.ci`` names for static functions are path-qualified
	(``"/abs/path/file.c:func"``) and clone-suffixed (``func.isra.0``) while
	``.su`` records use bare names (``func.isra``); both sides are reduced to
	``func`` so they meet in one key space.

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
	if "/" in name:
		name = name.rsplit(":", 1)[-1]
	return re.sub(r"(?:\.(?:isra|constprop|part|localalias|cold|lto_priv)(?:\.\d+)?)+$", "", name)


def _frames_by_bare_name(frames: Iterable[StackUsage]) -> Mapping[str, StackUsage]:
	"""Group ``.su`` records by bare name, keeping the largest frame.

	Duplicate names (weak stubs overridden by real implementations, or
	identically named statics across CUs) must not shadow the largest frame,
	or paths through the real function under-report.
	"""
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
