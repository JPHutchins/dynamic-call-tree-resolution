# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The deepest path from an entry."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from salix import Struct

from dynamic_call_tree_resolution.callgraph import CallPair
from dynamic_call_tree_resolution.stack_analysis.depths import (
	forward_depths,
	function_depth,
	preorder,
)
from dynamic_call_tree_resolution.stack_analysis.graph import (
	own_reasons,
)
from dynamic_call_tree_resolution.stack_analysis.reports import (
	frame_size,
	report_order,
	root_report,
)
from dynamic_call_tree_resolution.stack_analysis.values import (
	INDIRECT_CALLEE,
	MAX_CYCLE_SIZE,
	PathStep,
	Reason,
	StackGraph,
	frame_key,
)

if TYPE_CHECKING:
	from collections.abc import Mapping

	from dynamic_call_tree_resolution.callgraph import EdgeKind
	from dynamic_call_tree_resolution.stack_analysis.reports import ReportOrder


class _Forward(Struct):
	"""A large component's depths along the preorder from where a path entered it."""

	depths: Mapping[str, int]
	position: Mapping[str, int]


def deepest_path(graph: StackGraph, entry: str) -> tuple[PathStep, ...]:
	root = min(
		(node for node in graph.roots if frame_key(node) == entry),
		key=partial(_root_rank, graph),
		default=None,
	)
	if root is None:
		raise ValueError(f"{entry} is not an entry of the call graph")
	return _walk(graph, root, frozenset(), _entered(graph, root), 0, frozenset(), {})


class _RootRank(Struct, order=True):
	"""Where a root sorts among the roots of one entry."""

	order: ReportOrder
	root: str


class _CalleeDepth(Struct):
	"""A callee the walk may take next, with the depth below it."""

	callee: str
	depth: int


def _root_rank(graph: StackGraph, root: str) -> _RootRank:
	return _RootRank(order=report_order(root_report(graph, root)), root=root)


def _walk(
	graph: StackGraph,
	node: str,
	path: frozenset[str],
	forward: _Forward | None,
	cumulative: int,
	edge: frozenset[EdgeKind],
	within: dict[tuple[str, frozenset[str]], int],
) -> tuple[PathStep, ...]:
	frame = frame_size(graph, node)
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
	enters = callee.callee not in _members(graph, node)
	return (
		step,
		*_walk(
			graph,
			callee.callee,
			frozenset() if enters else path | {node},
			_entered(graph, callee.callee) if enters else forward,
			cumulative + frame,
			graph.kinds[CallPair(caller=node, callee=callee.callee)],
			within,
		),
	)


def _callee_depths(
	graph: StackGraph,
	node: str,
	path: frozenset[str],
	forward: _Forward | None,
	within: dict[tuple[str, frozenset[str]], int],
) -> tuple[_CalleeDepth, ...]:
	members = _members(graph, node)
	callees = sorted(graph.adjacency.get(node, frozenset()) - {INDIRECT_CALLEE})
	if forward is not None:
		return tuple(
			_CalleeDepth(
				callee=callee,
				depth=graph.depths[callee] if callee not in members else forward.depths[callee],
			)
			for callee in callees
			if callee not in members or forward.position[callee] > forward.position[node]
		)
	return tuple(
		_CalleeDepth(
			callee=callee,
			depth=graph.depths[callee]
			if callee not in members
			else function_depth(
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


def _callee_depth(callee: _CalleeDepth) -> int:
	return callee.depth


def _entered(graph: StackGraph, node: str) -> _Forward | None:
	members = _members(graph, node)
	if len(members) <= MAX_CYCLE_SIZE:
		return None
	order = preorder(node, members, graph.adjacency)
	return _Forward(
		depths=forward_depths(order, members, graph.adjacency, graph.frame_by_name, graph.depths),
		position={member: index for index, member in enumerate(order)},
	)


def _members(graph: StackGraph, node: str) -> frozenset[str]:
	return graph.components.get(node, frozenset({node}))


def _flags(graph: StackGraph, node: str) -> frozenset[Reason]:
	return frozenset(
		function_reason.reason
		for function_reason in own_reasons(
			node, graph.adjacency, graph.frame_by_name, graph.assumed_no_recursion, graph.narrowings
		)
	) | (
		frozenset({Reason.RECURSION})
		if len(_members(graph, node)) > 1 or node in graph.adjacency.get(node, frozenset())
		else frozenset[Reason]()
	)
