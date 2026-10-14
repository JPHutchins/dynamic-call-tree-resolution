# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Expanding indirect calls, and the call edges a thread's entry reaches."""

from __future__ import annotations

from collections import Counter
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Final

from salix import Struct, replace

from dynamic_call_tree_resolution.callgraph import CallEdge, EdgeKind
from dynamic_call_tree_resolution.stack_analysis.values import (
	INDIRECT_CALLEE,
	NO_CALLEE,
	Frame,
	frame_key,
)

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Mapping


_TRUSTED: Final = frozenset({EdgeKind.EXTERNAL, EdgeKind.DISPATCH})


def expand_indirect_calls(
	edges: Iterable[CallEdge],
	targets_by_caller: Mapping[str, frozenset[str]],
	fallback: frozenset[str],
	*,
	exact: bool = False,
	narrowed_by_caller: Mapping[str, Mapping[str, EdgeKind]] | None = None,
) -> tuple[CallEdge, ...]:
	edges_tuple = tuple(edges)
	graph_targets = partial(_graph_names, _raw_names_by_key(edges_tuple))
	return tuple(
		CallEdge(caller=edge.caller, callee=callee, kind=kind)
		for edge in edges_tuple
		for callee, kind in (
			_expansion(
				targets_by_caller.get(frame_key(edge.caller), frozenset[str]()),
				(narrowed_by_caller or {}).get(frame_key(edge.caller), {}),
				fallback,
				graph_targets,
				exact=exact,
			)
			or _unexpanded(
				targets_by_caller.get(frame_key(edge.caller), frozenset[str]()), edge.kind
			)
			if edge.callee == INDIRECT_CALLEE
			else ((edge.callee, edge.kind),)
		)
	)


def _expansion(
	targets: frozenset[str],
	narrowed: Mapping[str, EdgeKind],
	fallback: frozenset[str],
	graph_targets: Callable[[str], frozenset[str]],
	*,
	exact: bool,
) -> list[tuple[str, EdgeKind]]:
	candidates = targets - {INDIRECT_CALLEE}
	return sorted(
		(graph_target, _indirect_kind(target, candidates, narrowed))
		for target in (candidates - {NO_CALLEE})
		| (
			fallback
			if INDIRECT_CALLEE in targets or (not exact and not _complete(candidates, narrowed))
			else frozenset[str]()
		)
		for graph_target in graph_targets(target)
	)


def _complete(candidates: frozenset[str], narrowed: Mapping[str, EdgeKind]) -> bool:
	return bool(candidates) and all(
		target == NO_CALLEE or narrowed.get(target) in _TRUSTED for target in candidates
	)


def _unexpanded(targets: frozenset[str], kind: EdgeKind) -> list[tuple[str, EdgeKind]]:
	return [] if targets == frozenset({NO_CALLEE}) else [(INDIRECT_CALLEE, kind)]


def _indirect_kind(
	target: str, candidates: frozenset[str], narrowed: Mapping[str, EdgeKind]
) -> EdgeKind:
	return narrowed.get(target, EdgeKind.CANDIDATE if target in candidates else EdgeKind.FALLBACK)


def thread_edges(
	edges: Iterable[CallEdge],
	trampoline: str,
	entries_by_thread: Mapping[str, tuple[str, EdgeKind]],
) -> tuple[CallEdge, ...]:
	edges_tuple = tuple(edges)
	names_by_key = _raw_names_by_key(edges_tuple)
	return tuple(
		thread_edge
		for thread, (entry, started_by) in entries_by_thread.items()
		for thread_edge in (
			*(
				CallEdge(caller=thread, callee=edge.callee, kind=edge.kind)
				for edge in edges_tuple
				if frame_key(edge.caller) == trampoline and edge.callee != INDIRECT_CALLEE
			),
			*(
				CallEdge(caller=thread, callee=name, kind=started_by)
				for name in sorted(_graph_names(names_by_key, entry))
			),
		)
	)


class ThreadTargets(Struct):
	"""What one thread's own analysis resolved, by caller key."""

	reached: frozenset[str]
	targets_by_caller: Mapping[str, frozenset[str]]
	sites_by_caller: Mapping[str, int]
	narrowed_by_caller: Mapping[str, Mapping[str, EdgeKind]] = {}
	"""The targets each caller reaches only through a narrowing, with its edge."""


def own_thread_edges(
	edges: Iterable[CallEdge],
	trampoline: str,
	thread: str,
	entry: str,
	own: ThreadTargets,
	targets_by_caller: Mapping[str, frozenset[str]],
	fallback: frozenset[str],
	narrowed_by_caller: Mapping[str, Mapping[str, EdgeKind]] | None = None,
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
								own.narrowed_by_caller.get(frame_key(edge.caller), {}),
							),
						)
						for target in own.targets_by_caller[frame_key(edge.caller)] - {NO_CALLEE}
						for graph_target in graph_targets(target)
					)
					if frame_key(edge.caller) in complete
					else _expansion(
						targets_by_caller.get(frame_key(edge.caller), frozenset[str]()),
						(narrowed_by_caller or {}).get(frame_key(edge.caller), {}),
						fallback,
						graph_targets,
						exact=False,
					)
					or _unexpanded(
						targets_by_caller.get(frame_key(edge.caller), frozenset[str]()), edge.kind
					)
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


def thread_frames[F: Frame](
	frames: Iterable[F], trampoline: str, threads: Iterable[str]
) -> tuple[F, ...]:
	frames_tuple = tuple(frames)
	return tuple(
		replace(frame, function=thread)
		for thread in threads
		for frame in frames_tuple
		if frame_key(frame.function) == trampoline
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
