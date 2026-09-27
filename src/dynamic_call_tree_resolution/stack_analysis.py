# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Worst-case stack depth over a resolved call graph."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from salix import Struct

from dynamic_call_tree_resolution.callgraph import CallEdge
from dynamic_call_tree_resolution.stack_usage import StackUsage

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping


class StackReport(Struct):
	"""Worst-case stack depth of one entry point."""

	entry: str
	depth: int
	recursive: bool
	has_dynamic: bool


_MAX_CYCLE_SIZE = 20


def expand_indirect_calls(
	edges: Iterable[CallEdge],
	targets_by_caller: Mapping[str, frozenset[str]],
	fallback: frozenset[str],
) -> tuple[CallEdge, ...]:
	"""Replace GCC ``__indirect_call`` placeholders with resolved targets.

	Each caller's placeholders expand to the union of the candidates of the
	call sites extracted from that caller's code; callers without extracted
	sites expand to ``fallback`` (the union of all resolved targets). Both
	expansions are sound upper bounds for worst-case stack depth.
	"""
	return tuple(
		CallEdge(caller=edge.caller, callee=callee)
		for edge in edges
		for callee in (
			sorted(targets_by_caller.get(_frame_key(edge.caller), fallback))
			if edge.callee == "__indirect_call"
			else (edge.callee,)
		)
	)


def worst_case_depths(
	edges: Iterable[CallEdge],
	frames: Iterable[StackUsage],
) -> tuple[StackReport, ...]:
	"""Compute worst-case stack depths for every entry point.

	Entry points are functions with no incoming call edges. Cycles
	(recursion) are broken at the back edge and flagged. Frames using
	``alloca``/VLAs are counted but flagged.
	"""
	frame_by_name = _frames_by_bare_name(frames)
	callees_by_name: dict[str, set[str]] = {}
	for edge in edges:
		callees_by_name.setdefault(edge.caller, set()).add(edge.callee)
	adjacency = {caller: frozenset(callees) for caller, callees in callees_by_name.items()}
	depths = _depths_by_component(adjacency, frame_by_name)
	roots = sorted(
		adjacency.keys() - {callee for callees in adjacency.values() for callee in callees}
	)
	return tuple(
		report
		for report in sorted(
			(depths[entry] for entry in roots),
			key=lambda report: report.depth,
			reverse=True,
		)
	)


def _depths_by_component(
	adjacency: Mapping[str, frozenset[str]],
	frame_by_name: Mapping[str, StackUsage],
) -> Mapping[str, StackReport]:
	"""Per-node depth reports, computed one SCC at a time, sinks first.

	Tarjan emits strongly connected components in reverse topological
	order of the condensation, so every callee outside a node's own
	component is already computed when the node is reached. Only paths
	within one component need the path-based recursion, which keeps the
	search polynomial on diamond-heavy call graphs.

	Raises:
		ValueError: when a cycle exceeds ``_MAX_CYCLE_SIZE`` functions.
	"""
	memo: dict[str, StackReport] = {}
	components = _strongly_connected_components(adjacency)
	if any(len(component) > _MAX_CYCLE_SIZE for component in components):
		raise ValueError(
			f"call graph contains a cycle of "
			f"{max(len(component) for component in components)} functions"
		)
	known = {node for component in components for node in component}
	all_nodes = set(adjacency) | {callee for callees in adjacency.values() for callee in callees}
	for component in (*components, *((node,) for node in sorted(all_nodes - known))):
		in_component = frozenset(component)
		within: dict[tuple[str, frozenset[str]], StackReport] = {}
		for node in component:
			memo[node] = _depth(
				node, adjacency, frame_by_name, frozenset(), memo, in_component, within
			)
	return memo


def _depth(
	function: str,
	adjacency: Mapping[str, frozenset[str]],
	frame_by_name: Mapping[str, StackUsage],
	path: frozenset[str],
	memo: Mapping[str, StackReport],
	in_component: frozenset[str],
	within: dict[tuple[str, frozenset[str]], StackReport],
) -> StackReport:
	key = (function, path & in_component)
	if key in within:
		return within[key]
	frame = frame_by_name.get(_frame_key(function))
	current_path = path | {function}
	children: list[StackReport] = []
	for callee in sorted(adjacency.get(function, frozenset())):
		if callee in current_path:
			continue
		cached = None if callee in in_component else memo.get(callee)
		children.append(
			cached
			if cached is not None
			else _depth(callee, adjacency, frame_by_name, current_path, memo, in_component, within)
		)
	deepest = max(
		children,
		key=lambda report: report.depth,
		default=StackReport(entry=function, depth=0, recursive=False, has_dynamic=False),
	)
	report = StackReport(
		entry=_frame_key(function),
		depth=(frame.bytes if frame is not None else 0) + deepest.depth,
		recursive=deepest.recursive
		or any(callee in current_path for callee in adjacency.get(function, frozenset())),
		has_dynamic=(frame.dynamic if frame is not None else False) or deepest.has_dynamic,
	)
	within[key] = report
	return report


def _strongly_connected_components(
	adjacency: Mapping[str, frozenset[str]],
) -> tuple[tuple[str, ...], ...]:
	"""SCCs in reverse topological order of the condensation (sinks first)."""
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
		for neighbor in adjacency.get(node, frozenset()):
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

	for node in adjacency:
		if node not in indices:
			strong_connect(node)
	return tuple(components)


def _frame_key(name: str) -> str:
	"""Reduce a VCG or ``.su`` function name to its bare assembly name.

	GCC's ``.ci`` names for static functions are path-qualified
	(``"/abs/path/file.c:func"``) and clone-suffixed (``func.isra.0``) while
	``.su`` records use bare names (``func.isra``); both sides are reduced to
	``func`` so they meet in one key space.
	"""
	if "/" in name:
		name = name.rsplit(":", 1)[-1]
	return re.sub(r"\.(?:isra|constprop|part)(?:\.\d+)?$", "", name)


def _frames_by_bare_name(frames: Iterable[StackUsage]) -> Mapping[str, StackUsage]:
	"""Group ``.su`` records by bare name, keeping the largest frame.

	Duplicate names (weak stubs overridden by real implementations, or
	identically named statics across CUs) must not shadow the largest frame,
	or paths through the real function under-report.
	"""
	return {key: _merge_frame_records(records) for key, records in _grouped_frames(frames).items()}


def _grouped_frames(frames: Iterable[StackUsage]) -> dict[str, list[StackUsage]]:
	grouped: dict[str, list[StackUsage]] = {}
	for usage in frames:
		grouped.setdefault(_frame_key(usage.function), []).append(usage)
	return grouped


def _merge_frame_records(records: list[StackUsage]) -> StackUsage:
	largest = max(records, key=lambda usage: usage.bytes)
	return StackUsage(
		function=largest.function,
		bytes=largest.bytes,
		dynamic=any(usage.dynamic for usage in records),
	)
