# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Worst-case stack depth over a resolved call graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

from salix import Struct

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping

	from dynamic_call_tree_resolution.callgraph import CallEdge
	from dynamic_call_tree_resolution.stack_usage import StackUsage


class StackReport(Struct):
	"""Worst-case stack depth of one entry point."""

	entry: str
	depth: int
	recursive: bool
	has_dynamic: bool


def worst_case_depths(
	edges: Iterable[CallEdge],
	frames: Iterable[StackUsage],
) -> tuple[StackReport, ...]:
	"""Compute worst-case stack depths for every entry point.

	Entry points are functions with no incoming call edges. Cycles
	(recursion) are broken at the back edge and flagged. Frames using
	``alloca``/VLAs are counted but flagged.
	"""
	frame_by_name: Mapping[str, StackUsage] = {
		usage.function: usage for usage in sorted(frames, key=lambda usage: usage.function)
	}
	callees_by_name: dict[str, set[str]] = {}
	for edge in edges:
		callees_by_name.setdefault(edge.caller, set()).add(edge.callee)
	adjacency = {caller: frozenset(callees) for caller, callees in callees_by_name.items()}
	roots = sorted(
		adjacency.keys() - {callee for callees in adjacency.values() for callee in callees}
	)
	return tuple(
		report
		for report in sorted(
			(_depth(entry, adjacency, frame_by_name, frozenset()) for entry in roots),
			key=lambda report: report.depth,
			reverse=True,
		)
	)


def _depth(
	function: str,
	adjacency: Mapping[str, frozenset[str]],
	frame_by_name: Mapping[str, StackUsage],
	path: frozenset[str],
) -> StackReport:
	frame = frame_by_name.get(function)
	current_path = path | {function}
	descendants = (
		_depth(callee, adjacency, frame_by_name, current_path)
		for callee in sorted(adjacency.get(function, frozenset()))
		if callee not in current_path
	)
	deepest = max(
		descendants,
		key=lambda report: report.depth,
		default=StackReport(entry=function, depth=0, recursive=False, has_dynamic=False),
	)
	return StackReport(
		entry=function,
		depth=(frame.bytes if frame is not None else 0) + deepest.depth,
		recursive=deepest.recursive
		or any(callee in current_path for callee in adjacency.get(function, frozenset())),
		has_dynamic=(frame.dynamic if frame is not None else False) or deepest.has_dynamic,
	)
