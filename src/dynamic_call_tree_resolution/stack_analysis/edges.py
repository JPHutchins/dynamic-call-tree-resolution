# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Expanding indirect calls, and the call edges a thread's entry reaches."""

from __future__ import annotations

from collections import Counter
from functools import partial
from itertools import chain, groupby, product
from typing import TYPE_CHECKING, Final

from salix import Struct

from dynamic_call_tree_resolution.callgraph import CallEdge, EdgeKind
from dynamic_call_tree_resolution.composition import flatmap
from dynamic_call_tree_resolution.stack_analysis.values import (
	INDIRECT_CALLEE,
	NO_CALLEE,
	Frame,
	frame_key,
)
from dynamic_call_tree_resolution.structs import replace_as_given

if TYPE_CHECKING:
	from collections.abc import Callable, Collection, Iterable, Iterator, Mapping


_TRUSTED: Final = frozenset({EdgeKind.EXTERNAL, EdgeKind.DISPATCH})


class ThreadEntry(Struct):
	"""Where a thread starts, and the kind of edge that starts it."""

	entry: str
	started_by: EdgeKind


class _Callee(Struct, order=True):
	"""A callee an edge leads to, with the edge's kind."""

	callee: str
	kind: EdgeKind


def expand_indirect_calls(
	edges: Iterable[CallEdge],
	targets_by_caller: Mapping[str, frozenset[str]],
	fallback: frozenset[str],
	*,
	exact: bool = False,
	narrowed_by_caller: Mapping[str, Mapping[str, EdgeKind]] | None = None,
	complete: frozenset[str] = frozenset(),
) -> tuple[CallEdge, ...]:
	edges_tuple = tuple(edges)
	graph_targets = partial(_graph_names, _raw_names_by_key(edges_tuple))
	return tuple(
		flatmap(
			lambda edge: _edges_to(
				edge.caller,
				_expanded_callees(
					edge,
					targets_by_caller,
					narrowed_by_caller,
					fallback,
					graph_targets,
					exact=exact or frame_key(edge.caller) in complete,
				),
			),
			edges_tuple,
		)
	)


def _expanded_callees(
	edge: CallEdge,
	targets_by_caller: Mapping[str, frozenset[str]],
	narrowed_by_caller: Mapping[str, Mapping[str, EdgeKind]] | None,
	fallback: frozenset[str],
	graph_targets: Callable[[str], frozenset[str]],
	*,
	exact: bool,
) -> Iterable[_Callee]:
	targets = targets_by_caller.get(frame_key(edge.caller), frozenset[str]())
	return (
		_expansion(
			targets,
			(narrowed_by_caller or {}).get(frame_key(edge.caller), {}),
			fallback,
			graph_targets,
			exact=exact,
		)
		or _unexpanded(targets, edge.kind)
		if edge.callee == INDIRECT_CALLEE
		else (_Callee(callee=edge.callee, kind=edge.kind),)
	)


def _edges_to(caller: str, callees: Iterable[_Callee]) -> Iterator[CallEdge]:
	return (CallEdge(caller=caller, callee=callee.callee, kind=callee.kind) for callee in callees)


def _kinded(names: Iterable[str], kind: EdgeKind) -> Iterator[_Callee]:
	return (_Callee(callee=name, kind=kind) for name in names)


def _expansion(
	targets: frozenset[str],
	narrowed: Mapping[str, EdgeKind],
	fallback: frozenset[str],
	graph_targets: Callable[[str], frozenset[str]],
	*,
	exact: bool,
) -> list[_Callee]:
	candidates = targets - {INDIRECT_CALLEE}
	to_expand = (candidates - {NO_CALLEE}) | (
		fallback
		if INDIRECT_CALLEE in targets or (not exact and not _complete(candidates, narrowed))
		else frozenset[str]()
	)
	return sorted(
		flatmap(
			lambda target: _kinded(
				graph_targets(target), _indirect_kind(target, candidates, narrowed)
			),
			to_expand,
		)
	)


def _complete(candidates: frozenset[str], narrowed: Mapping[str, EdgeKind]) -> bool:
	return bool(candidates) and all(
		target == NO_CALLEE or narrowed.get(target) in _TRUSTED for target in candidates
	)


def _unexpanded(targets: frozenset[str], kind: EdgeKind) -> list[_Callee]:
	return [] if targets == frozenset({NO_CALLEE}) else [_Callee(callee=INDIRECT_CALLEE, kind=kind)]


def _indirect_kind(
	target: str, candidates: frozenset[str], narrowed: Mapping[str, EdgeKind]
) -> EdgeKind:
	return narrowed.get(target, EdgeKind.CANDIDATE if target in candidates else EdgeKind.FALLBACK)


def thread_edges(
	edges: Iterable[CallEdge],
	trampoline: str,
	entries_by_thread: Mapping[str, ThreadEntry],
) -> tuple[CallEdge, ...]:
	edges_tuple = tuple(edges)
	names_by_key = _raw_names_by_key(edges_tuple)
	return tuple(
		flatmap(
			lambda thread: _entry_edges(
				edges_tuple, trampoline, names_by_key, thread, entries_by_thread[thread]
			),
			entries_by_thread,
		)
	)


def _entry_edges(
	edges: Collection[CallEdge],
	trampoline: str,
	names_by_key: Mapping[str, frozenset[str]],
	thread: str,
	started: ThreadEntry,
) -> Iterator[CallEdge]:
	return chain(
		(
			CallEdge(caller=thread, callee=edge.callee, kind=edge.kind)
			for edge in edges
			if frame_key(edge.caller) == trampoline and edge.callee != INDIRECT_CALLEE
		),
		(
			CallEdge(caller=thread, callee=name, kind=started.started_by)
			for name in sorted(_graph_names(names_by_key, started.entry))
		),
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
	complete: frozenset[str] = frozenset(),
) -> tuple[CallEdge, ...]:
	edges_tuple = tuple(edges)
	graph_targets = partial(_graph_names, _raw_names_by_key(edges_tuple))
	node = partial(_thread_node, thread, own.reached)
	own_complete = complete_callers(edges_tuple, own.targets_by_caller, own.sites_by_caller)

	def callees(edge: CallEdge) -> Iterable[_Callee]:
		return (
			(
				_own_expansion(own, node, graph_targets, frame_key(edge.caller))
				if frame_key(edge.caller) in own_complete
				else _expanded_callees(
					edge,
					targets_by_caller,
					narrowed_by_caller,
					fallback,
					graph_targets,
					exact=frame_key(edge.caller) in complete,
				)
			)
			if edge.callee == INDIRECT_CALLEE
			else (_Callee(callee=node(edge.callee), kind=edge.kind),)
		)

	return tuple(
		chain(
			(
				CallEdge(caller=thread, callee=node(edge.callee), kind=edge.kind)
				for edge in edges_tuple
				if frame_key(edge.caller) == trampoline and edge.callee != INDIRECT_CALLEE
			),
			(
				CallEdge(caller=thread, callee=node(name), kind=EdgeKind.THREAD)
				for name in sorted(graph_targets(entry))
			),
			flatmap(
				lambda edge: _edges_to(_in_thread(thread, edge.caller), callees(edge)),
				(edge for edge in edges_tuple if frame_key(edge.caller) in own.reached),
			),
		)
	)


def _own_expansion(
	own: ThreadTargets,
	node: Callable[[str], str],
	graph_targets: Callable[[str], frozenset[str]],
	caller_key: str,
) -> list[_Callee]:
	return sorted(
		flatmap(
			lambda target: _kinded(
				map(node, graph_targets(target)),
				_indirect_kind(
					target,
					own.targets_by_caller[caller_key],
					own.narrowed_by_caller.get(caller_key, {}),
				),
			),
			own.targets_by_caller[caller_key] - {NO_CALLEE},
		)
	)


def complete_callers(
	edges: Iterable[CallEdge],
	targets_by_caller: Mapping[str, frozenset[str]],
	sites_by_caller: Mapping[str, int],
) -> frozenset[str]:
	"""The callers whose indirect calls expand to their targets alone, without the fallback."""
	indirect_edges = Counter(
		frame_key(edge.caller) for edge in edges if edge.callee == INDIRECT_CALLEE
	)
	return frozenset(
		caller
		for caller, targets in targets_by_caller.items()
		if INDIRECT_CALLEE not in targets
		and sites_by_caller.get(caller, 0) >= indirect_edges[caller]
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
	trampoline_frames = tuple(frame for frame in frames if frame_key(frame.function) == trampoline)
	return tuple(
		replace_as_given(frame, function=thread)
		for thread, frame in product(threads, trampoline_frames)
	)


def _raw_names_by_key(edges: tuple[CallEdge, ...]) -> Mapping[str, frozenset[str]]:
	return {
		key: frozenset(group)
		for key, group in groupby(
			sorted(set(flatmap(lambda edge: (edge.caller, edge.callee), edges)), key=frame_key),
			key=frame_key,
		)
	}


def _graph_names(names_by_key: Mapping[str, frozenset[str]], target: str) -> frozenset[str]:
	return names_by_key.get(frame_key(target), frozenset({target}))
