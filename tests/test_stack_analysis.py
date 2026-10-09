# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Tests for :mod:`dynamic_call_tree_resolution.stack_analysis`."""

import os
import subprocess
import sys
from itertools import pairwise
from typing import TYPE_CHECKING

import pytest

from dynamic_call_tree_resolution import (
	Bounded,
	CallEdge,
	EdgeKind,
	PathStep,
	Reason,
	StackReport,
	StackUsage,
	Unbounded,
	deepest_path,
	expand_indirect_calls,
	load,
	load_callgraph,
	load_stack_usages,
	per_caller_candidates,
	resolve,
	stack_graph,
	stack_reports,
	worst_case_depths,
)
from dynamic_call_tree_resolution.model import Address, RtosModel, SystemThread
from dynamic_call_tree_resolution.stack_analysis import (
	INDIRECT_CALLEE,
	NO_CALLEE,
	Handled,
	LevelSource,
	Nested,
	Nesting,
	PriorityLevels,
	ThreadTargets,
	entry_report,
	frame_key,
	interrupt_stack_report,
	own_thread_edges,
	thread_edges,
	thread_frames,
	thread_reports,
)

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path

ARTIFACT_DIRECTORIES = ("counter-su", "sensor-two-impl")

FRAMES = (
	StackUsage(function="main", bytes=8, bounded=True),
	StackUsage(function="a", bytes=16, bounded=True),
	StackUsage(function="b", bytes=24, bounded=True),
	StackUsage(function="c", bytes=32, bounded=True),
	StackUsage(function="leaf", bytes=4, bounded=True),
)


def test_a_thread_is_the_trampolines_frame_and_calls_with_its_indirect_call_to_the_entry() -> None:
	edges = (
		CallEdge(caller="trampoline", callee=INDIRECT_CALLEE),
		CallEdge(caller="trampoline", callee="abort"),
		CallEdge(caller="/src/app.c:entry", callee="leaf"),
	)
	frames = (
		StackUsage(function="trampoline", bytes=8, bounded=True),
		StackUsage(function="abort", bytes=50, bounded=True),
		StackUsage(function="entry", bytes=100, bounded=True),
		StackUsage(function="leaf", bytes=10, bounded=True),
	)
	threads = thread_edges(edges, "trampoline", {"worker_tid": ("entry", EdgeKind.THREAD)})
	depths = {
		report.entry: report.bound
		for report in worst_case_depths(
			(*edges, *threads), (*frames, *thread_frames(frames, "trampoline", ("worker_tid",)))
		)
	}
	assert (threads, depths.get("worker_tid"), "entry" in depths) == (
		(
			CallEdge(caller="worker_tid", callee="abort"),
			CallEdge(caller="worker_tid", callee="/src/app.c:entry", kind=EdgeKind.THREAD),
		),
		Bounded(bytes=118),
		False,
	)


def test_a_system_threads_row_adds_what_the_rtos_model_adds_to_the_whole_images_tree() -> None:
	assert thread_reports(
		(
			StackReport(entry="main_thread", bound=Bounded(bytes=8 + 40)),
			StackReport(entry="isr", bound=Bounded(bytes=24)),
		),
		{},
		RtosModel(
			name="test",
			evidence=(),
			threads=(),
			trampoline="trampoline",
			stack_reservation=16,
			exception_frame=36,
			system_threads=(SystemThread(name="main_thread", entry=Address(0x100)),),
		),
	) == (
		StackReport(
			entry="main_thread",
			bound=Bounded(bytes=16 + 8 + 40 + 36, stack_reservation=16, exception_frame=36),
		),
		StackReport(entry="isr", bound=Bounded(bytes=24)),
	)


def _handler(exception: int, name: str, depth: int) -> Handled:
	return Handled(exception=exception, report=StackReport(entry=name, bound=Bounded(bytes=depth)))


def test_an_interrupt_stack_nests_the_deepest_configurable_handlers_that_fit_then_the_fixed() -> (
	None
):
	levels = PriorityLevels(count=3, source=LevelSource.DEVICETREE)
	assert interrupt_stack_report(
		"interrupts",
		_handler(1, "reset", 100),
		(_handler(3, "hard_fault", 20), _handler(2, "nmi", 8)),
		(
			_handler(4, "fault", 20),
			_handler(5, "fault", 20),
			_handler(16, "isr", 50),
			_handler(17, "isr", 50),
			_handler(11, "svc", 10),
		),
		levels=levels,
		exception_frame=36,
	) == StackReport(
		entry="interrupts",
		bound=Bounded(bytes=100 + 2 * (36 + 50) + (36 + 20) + (36 + 20) + (36 + 8)),
		nesting=Nesting(
			base=Nested(exception=1, handler="reset", depth=100),
			chain=(
				Nested(exception=16, handler="isr", depth=50),
				Nested(exception=17, handler="isr", depth=50),
				Nested(exception=4, handler="fault", depth=20),
				Nested(exception=3, handler="hard_fault", depth=20),
				Nested(exception=2, handler="nmi", depth=8),
			),
			levels=levels,
			exception_frame=36,
		),
	)


def test_an_interrupt_stack_is_unbounded_when_a_handler_outside_its_chain_is() -> None:
	assert interrupt_stack_report(
		"interrupts",
		_handler(1, "reset", 100),
		(),
		(
			_handler(16, "isr", 50),
			Handled(
				exception=11,
				report=StackReport(
					entry="svc",
					bound=Unbounded(
						at_least=4,
						recursion=frozenset({"svc"}),
						unmeasured=frozenset(),
						dynamic=frozenset(),
						unresolved=frozenset(),
					),
				),
			),
		),
		levels=PriorityLevels(count=1, source=LevelSource.VECTOR_TABLE),
		exception_frame=36,
	).bound == Unbounded(
		at_least=100 + 36 + 50,
		recursion=frozenset({"svc"}),
		unmeasured=frozenset(),
		dynamic=frozenset(),
		unresolved=frozenset(),
	)


def test_an_entry_report_reads_a_node_that_is_not_a_root_and_a_function_the_graph_lacks() -> None:
	graph = stack_graph(
		(CallEdge(caller="main", callee="handler"),),
		(
			StackUsage(function="main", bytes=8, bounded=True),
			StackUsage(function="handler", bytes=24, bounded=True),
		),
	)
	assert (entry_report(graph, "handler"), entry_report(graph, "missing")) == (
		StackReport(entry="handler", bound=Bounded(bytes=24)),
		StackReport(
			entry="missing",
			bound=Unbounded(
				at_least=0,
				recursion=frozenset(),
				unmeasured=frozenset({"missing"}),
				dynamic=frozenset(),
				unresolved=frozenset(),
			),
		),
	)


SHARED_ENTRY_EDGES = (
	CallEdge(caller="trampoline", callee=INDIRECT_CALLEE),
	CallEdge(caller="trampoline", callee="abort"),
	CallEdge(caller="/src/app.c:entry", callee=INDIRECT_CALLEE),
)

SHARED_ENTRY_FRAMES = (
	StackUsage(function="trampoline", bytes=8, bounded=True),
	StackUsage(function="abort", bytes=50, bounded=True),
	StackUsage(function="entry", bytes=100, bounded=True),
	StackUsage(function="cheap", bytes=10, bounded=True),
	StackUsage(function="costly", bytes=1000, bounded=True),
)


def _own_thread_depth(
	own: ThreadTargets, edges: tuple[CallEdge, ...]
) -> tuple[Bounded | Unbounded, tuple[tuple[str, frozenset[EdgeKind]], ...]]:
	graph = stack_graph(
		(
			*expand_indirect_calls(
				edges,
				{"entry": frozenset({"cheap", "costly"})},
				frozenset({"entry", "cheap", "costly"}),
			),
			*own_thread_edges(
				edges,
				"trampoline",
				"worker_tid",
				"entry",
				own,
				{"entry": frozenset({"cheap", "costly"})},
				frozenset({"entry", "cheap", "costly"}),
			),
		),
		(
			*SHARED_ENTRY_FRAMES,
			*thread_frames(SHARED_ENTRY_FRAMES, "trampoline", ("worker_tid",)),
		),
	)
	return (
		next(report for report in stack_reports(graph) if report.entry == "worker_tid").bound,
		tuple((step.function, step.edge) for step in deepest_path(graph, "worker_tid")),
	)


@pytest.mark.parametrize(
	("own", "edges"),
	[
		pytest.param(
			ThreadTargets(
				reached=frozenset({"trampoline", "entry", "cheap"}),
				targets_by_caller={
					"trampoline": frozenset({"entry"}),
					"entry": frozenset({"cheap"}),
				},
				sites_by_caller={"trampoline": 1, "entry": 1},
			),
			SHARED_ENTRY_EDGES,
			id="resolved",
		),
	],
)
def test_a_thread_follows_the_sites_its_own_analysis_resolved(
	own: ThreadTargets, edges: tuple[CallEdge, ...]
) -> None:
	assert _own_thread_depth(own, edges) == (
		Bounded(bytes=8 + 100 + 10),
		(
			("worker_tid", frozenset()),
			("entry", frozenset({EdgeKind.THREAD})),
			("cheap", frozenset({EdgeKind.CANDIDATE})),
		),
	)


def test_no_image_node_calls_into_a_threads_own_nodes() -> None:
	edges = (
		*expand_indirect_calls(
			SHARED_ENTRY_EDGES,
			{"entry": frozenset({"cheap", "costly"})},
			frozenset({"entry", "cheap", "costly"}),
		),
		*own_thread_edges(
			SHARED_ENTRY_EDGES,
			"trampoline",
			"worker_tid",
			"entry",
			ThreadTargets(
				reached=frozenset({"trampoline", "entry", "cheap"}),
				targets_by_caller={
					"trampoline": frozenset({"entry"}),
					"entry": frozenset({"cheap"}),
				},
				sites_by_caller={"trampoline": 1, "entry": 1},
			),
			{"entry": frozenset({"cheap", "costly"})},
			frozenset({"entry", "cheap", "costly"}),
		),
	)
	assert (
		sorted(edge.callee for edge in edges if edge.callee.startswith("worker_tid/:")),
		[
			edge
			for edge in edges
			if edge.callee.startswith("worker_tid/:")
			and edge.caller != "worker_tid"
			and not edge.caller.startswith("worker_tid/:")
		],
	) == (["worker_tid/:/src/app.c:entry", "worker_tid/:/src/app.c:entry", "worker_tid/:cheap"], [])


@pytest.mark.parametrize(
	("own", "edges"),
	[
		pytest.param(
			ThreadTargets(
				reached=frozenset({"trampoline", "entry"}),
				targets_by_caller={
					"trampoline": frozenset({"entry"}),
					"entry": frozenset({INDIRECT_CALLEE}),
				},
				sites_by_caller={"trampoline": 1, "entry": 1},
			),
			SHARED_ENTRY_EDGES,
			id="unresolved-site",
		),
		pytest.param(
			ThreadTargets(
				reached=frozenset({"trampoline", "entry", "cheap"}),
				targets_by_caller={
					"trampoline": frozenset({"entry"}),
					"entry": frozenset({"cheap"}),
				},
				sites_by_caller={"trampoline": 1, "entry": 1},
			),
			(*SHARED_ENTRY_EDGES, CallEdge(caller="/src/app.c:entry", callee=INDIRECT_CALLEE)),
			id="fewer-sites-than-the-callgraph",
		),
	],
)
def test_a_thread_takes_the_global_expansion_of_a_site_its_own_analysis_did_not_resolve(
	own: ThreadTargets, edges: tuple[CallEdge, ...]
) -> None:
	assert _own_thread_depth(own, edges) == (
		Unbounded(
			at_least=8 + 100 + 100 + 1000,
			recursion=frozenset({"entry"}),
			unmeasured=frozenset(),
			dynamic=frozenset(),
			unresolved=frozenset(),
		),
		(
			("worker_tid", frozenset()),
			("entry", frozenset({EdgeKind.THREAD})),
			("entry", frozenset({EdgeKind.FALLBACK})),
			("costly", frozenset({EdgeKind.CANDIDATE})),
		),
	)


def _cycle(at_least: int, recursion: frozenset[str]) -> Unbounded:
	return Unbounded(
		at_least=at_least,
		recursion=recursion,
		unmeasured=frozenset(),
		dynamic=frozenset(),
		unresolved=frozenset(),
	)


@pytest.mark.parametrize(
	("assumed", "bounds"),
	[
		pytest.param(
			frozenset[str](),
			{
				"main": _cycle(8 + 16 + 4, frozenset({"a"})),
				"b": _cycle(24 + 32 + 8, frozenset({"x", "y"})),
			},
			id="none",
		),
		pytest.param(
			frozenset({"a", "x"}),
			{
				"main": Bounded(bytes=8 + 16 + 4, assumed_no_recursion=frozenset({"a"})),
				"b": _cycle(24 + 32 + 8, frozenset({"x", "y"})),
			},
			id="self-call-cut-and-two-function-cycle-kept",
		),
	],
)
def test_assuming_a_function_never_calls_itself_drops_only_its_self_call(
	assumed: frozenset[str], bounds: Mapping[str, Bounded | Unbounded]
) -> None:
	graph = stack_graph(
		(
			CallEdge(caller="main", callee="a"),
			CallEdge(caller="a", callee="a"),
			CallEdge(caller="a", callee="leaf"),
			CallEdge(caller="b", callee="x"),
			CallEdge(caller="x", callee="y"),
			CallEdge(caller="y", callee="x"),
		),
		(
			StackUsage(function="main", bytes=8, bounded=True),
			StackUsage(function="a", bytes=16, bounded=True),
			StackUsage(function="leaf", bytes=4, bounded=True),
			StackUsage(function="b", bytes=24, bounded=True),
			StackUsage(function="x", bytes=32, bounded=True),
			StackUsage(function="y", bytes=8, bounded=True),
		),
		assumed_no_recursion=assumed,
	)
	assert {
		report.entry: report.bound for report in stack_reports(graph) if report.entry in bounds
	} == bounds


def test_worst_case_depth_takes_the_deepest_branch() -> None:
	edges = (
		CallEdge(caller="main", callee="a"),
		CallEdge(caller="main", callee="b"),
		CallEdge(caller="a", callee="leaf"),
		CallEdge(caller="b", callee="c"),
		CallEdge(caller="c", callee="leaf"),
	)
	assert worst_case_depths(edges, FRAMES) == (
		StackReport(entry="main", bound=Bounded(bytes=8 + 24 + 32 + 4)),
	)


def test_recursion_leaves_the_depth_unbounded() -> None:
	edges = (
		CallEdge(caller="main", callee="a"),
		CallEdge(caller="a", callee="a"),
		CallEdge(caller="a", callee="leaf"),
	)
	frames = (
		StackUsage(function="main", bytes=8, bounded=True),
		StackUsage(function="a", bytes=16, bounded=True),
		StackUsage(function="leaf", bytes=4, bounded=True),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="main",
			bound=Unbounded(
				at_least=8 + 16 + 4,
				recursion=frozenset({"a"}),
				unmeasured=frozenset(),
				dynamic=frozenset(),
				unresolved=frozenset(),
			),
		),
	)


def test_an_unbounded_dynamic_frame_leaves_the_depth_unbounded() -> None:
	edges = (CallEdge(caller="main", callee="leaf"),)
	frames = (
		StackUsage(function="main", bytes=8, bounded=True),
		StackUsage(function="leaf", bytes=4, bounded=False),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="main",
			bound=Unbounded(
				at_least=12,
				recursion=frozenset(),
				unmeasured=frozenset(),
				dynamic=frozenset({"leaf"}),
				unresolved=frozenset(),
			),
		),
	)


def test_a_frame_without_a_record_leaves_the_depth_unbounded() -> None:
	edges = (CallEdge(caller="main", callee="no_record"),)
	frames = (StackUsage(function="main", bytes=8, bounded=True),)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="main",
			bound=Unbounded(
				at_least=8,
				recursion=frozenset(),
				unmeasured=frozenset({"no_record"}),
				dynamic=frozenset(),
				unresolved=frozenset(),
			),
		),
	)


def test_path_qualified_static_names_match_su_records() -> None:
	edges = (
		CallEdge(caller="main", callee="/home/jp/zephyr/kernel/timeslicing.c:slice_reset"),
		CallEdge(caller="/home/jp/zephyr/kernel/timeslicing.c:slice_reset", callee="leaf"),
	)
	frames = (
		StackUsage(function="main", bytes=8, bounded=True),
		StackUsage(function="slice_reset", bytes=64, bounded=True),
		StackUsage(function="leaf", bytes=4, bounded=True),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(entry="main", bound=Bounded(bytes=8 + 64 + 4)),
	)


def test_clone_suffixes_match_across_sources() -> None:
	edges = (CallEdge(caller="main", callee="k_sleep_ticks.isra.0"),)
	frames = (
		StackUsage(function="main", bytes=8, bounded=True),
		StackUsage(function="k_sleep_ticks.isra", bytes=32, bounded=True),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(entry="main", bound=Bounded(bytes=8 + 32)),
	)


def test_duplicate_su_names_keep_the_largest_frame() -> None:
	edges = (CallEdge(caller="bg_thread_main", callee="main"),)
	frames = (
		StackUsage(function="bg_thread_main", bytes=8, bounded=True),
		StackUsage(function="main", bytes=4, bounded=True),
		StackUsage(function="main", bytes=160, bounded=True),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(entry="bg_thread_main", bound=Bounded(bytes=8 + 160)),
	)


def test_duplicate_su_names_keep_any_unbounded_frame() -> None:
	edges = (CallEdge(caller="bg_thread_main", callee="main"),)
	frames = (
		StackUsage(function="bg_thread_main", bytes=8, bounded=True),
		StackUsage(function="main", bytes=160, bounded=True),
		StackUsage(function="main", bytes=4, bounded=False),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="bg_thread_main",
			bound=Unbounded(
				at_least=168,
				recursion=frozenset(),
				unmeasured=frozenset(),
				dynamic=frozenset({"main"}),
				unresolved=frozenset(),
			),
		),
	)


def test_static_entry_points_report_bare_names() -> None:
	edges = (CallEdge(caller="/home/jp/zephyr/main.c:static_entry", callee="leaf"),)
	frames = (
		StackUsage(function="static_entry", bytes=16, bounded=True),
		StackUsage(function="leaf", bytes=4, bounded=True),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(entry="static_entry", bound=Bounded(bytes=20)),
	)


def test_same_named_statics_in_different_files_stay_distinct_nodes() -> None:
	edges = (
		CallEdge(caller="main", callee="/home/jp/zephyr/a.c:helper"),
		CallEdge(caller="main", callee="/home/jp/zephyr/b.c:helper"),
		CallEdge(caller="/home/jp/zephyr/a.c:helper", callee="leaf_a"),
		CallEdge(caller="/home/jp/zephyr/b.c:helper", callee="leaf_b"),
	)
	frames = (
		StackUsage(function="main", bytes=8, bounded=True),
		StackUsage(function="helper", bytes=16, bounded=True),
		StackUsage(function="leaf_a", bytes=40, bounded=True),
		StackUsage(function="leaf_b", bytes=24, bounded=True),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(entry="main", bound=Bounded(bytes=8 + 16 + 40)),
	)


def test_reasons_propagate_from_non_deepest_branches() -> None:
	edges = (
		CallEdge(caller="main", callee="deep"),
		CallEdge(caller="main", callee="shallow"),
		CallEdge(caller="deep", callee="leaf"),
	)
	frames = (
		StackUsage(function="main", bytes=8, bounded=True),
		StackUsage(function="deep", bytes=40, bounded=True),
		StackUsage(function="shallow", bytes=4, bounded=False),
		StackUsage(function="leaf", bytes=4, bounded=True),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="main",
			bound=Unbounded(
				at_least=8 + 40 + 4,
				recursion=frozenset(),
				unmeasured=frozenset(),
				dynamic=frozenset({"shallow"}),
				unresolved=frozenset(),
			),
		),
	)


def test_recursion_propagates_from_non_deepest_branches() -> None:
	edges = (
		CallEdge(caller="main", callee="deep"),
		CallEdge(caller="main", callee="shallow"),
		CallEdge(caller="deep", callee="leaf"),
		CallEdge(caller="shallow", callee="shallow"),
	)
	frames = (
		StackUsage(function="main", bytes=8, bounded=True),
		StackUsage(function="deep", bytes=40, bounded=True),
		StackUsage(function="shallow", bytes=4, bounded=True),
		StackUsage(function="leaf", bytes=4, bounded=True),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="main",
			bound=Unbounded(
				at_least=8 + 40 + 4,
				recursion=frozenset({"shallow"}),
				unmeasured=frozenset(),
				dynamic=frozenset(),
				unresolved=frozenset(),
			),
		),
	)


def test_edgeless_su_functions_are_entry_points() -> None:
	edges = (CallEdge(caller="main", callee="leaf"),)
	frames = (
		StackUsage(function="main", bytes=8, bounded=True),
		StackUsage(function="leaf", bytes=4, bounded=True),
		StackUsage(function="isr", bytes=64, bounded=True),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(entry="isr", bound=Bounded(bytes=64)),
		StackReport(entry="main", bound=Bounded(bytes=12)),
	)


def test_an_unmeasured_frame_off_the_deepest_path_leaves_the_depth_unbounded() -> None:
	edges = (
		CallEdge(caller="main", callee="deep"),
		CallEdge(caller="main", callee="unknown_asm_handler"),
	)
	frames = (
		StackUsage(function="main", bytes=16, bounded=True),
		StackUsage(function="deep", bytes=100, bounded=True),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="main",
			bound=Unbounded(
				at_least=16 + 100,
				recursion=frozenset(),
				unmeasured=frozenset({"unknown_asm_handler"}),
				dynamic=frozenset(),
				unresolved=frozenset(),
			),
		),
	)


def test_mutual_recursion_names_every_function_on_the_cycle() -> None:
	edges = (
		CallEdge(caller="entry", callee="a"),
		CallEdge(caller="a", callee="b"),
		CallEdge(caller="b", callee="a"),
		CallEdge(caller="b", callee="leaf"),
	)
	frames = (
		StackUsage(function="entry", bytes=4, bounded=True),
		StackUsage(function="a", bytes=16, bounded=True),
		StackUsage(function="b", bytes=32, bounded=True),
		StackUsage(function="leaf", bytes=8, bounded=True),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="entry",
			bound=Unbounded(
				at_least=4 + 16 + 32 + 8,
				recursion=frozenset({"a", "b"}),
				unmeasured=frozenset(),
				dynamic=frozenset(),
				unresolved=frozenset(),
			),
		),
	)


def test_an_indirect_call_without_candidates_is_unresolved_not_unmeasured() -> None:
	edges = (
		CallEdge(caller="main", callee="__indirect_call"),
		CallEdge(caller="main", callee="leaf"),
	)
	frames = (
		StackUsage(function="main", bytes=8, bounded=True),
		StackUsage(function="leaf", bytes=4, bounded=True),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="main",
			bound=Unbounded(
				at_least=8 + 4,
				recursion=frozenset(),
				unmeasured=frozenset(),
				dynamic=frozenset(),
				unresolved=frozenset({"main"}),
			),
		),
	)


def test_unbounded_entries_come_before_deeper_bounded_ones() -> None:
	edges = (
		CallEdge(caller="bounded_main", callee="leaf"),
		CallEdge(caller="recursive_main", callee="helper"),
		CallEdge(caller="helper", callee="helper"),
	)
	frames = (
		StackUsage(function="bounded_main", bytes=100, bounded=True),
		StackUsage(function="leaf", bytes=4, bounded=True),
		StackUsage(function="recursive_main", bytes=8, bounded=True),
		StackUsage(function="helper", bytes=4, bounded=True),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="recursive_main",
			bound=Unbounded(
				at_least=8 + 4,
				recursion=frozenset({"helper"}),
				unmeasured=frozenset(),
				dynamic=frozenset(),
				unresolved=frozenset(),
			),
		),
		StackReport(entry="bounded_main", bound=Bounded(bytes=104)),
	)


def test_an_oversized_cycle_is_unbounded_recursion_over_a_real_path() -> None:
	ring = tuple(f"node_{index}" for index in range(65))
	edges = (
		CallEdge(caller="main", callee="node_0"),
		*(CallEdge(caller=node, callee=ring[(index + 1) % 65]) for index, node in enumerate(ring)),
		CallEdge(caller="node_0", callee="leaf"),
	)
	frames = (
		StackUsage(function="main", bytes=8, bounded=True),
		StackUsage(function="leaf", bytes=16, bounded=True),
		*(StackUsage(function=node, bytes=4, bounded=True) for node in ring),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="main",
			bound=Unbounded(
				at_least=8 + 65 * 4,
				recursion=frozenset(ring),
				unmeasured=frozenset(),
				dynamic=frozenset(),
				unresolved=frozenset(),
			),
		),
	)


def test_within_cycle_diamonds_reuse_computed_states() -> None:
	edges = (
		CallEdge(caller="main", callee="a"),
		CallEdge(caller="a", callee="b"),
		CallEdge(caller="a", callee="c"),
		CallEdge(caller="b", callee="c"),
		CallEdge(caller="c", callee="b"),
		CallEdge(caller="b", callee="d"),
		CallEdge(caller="c", callee="d"),
		CallEdge(caller="d", callee="a"),
	)
	frames = (
		StackUsage(function="main", bytes=2, bounded=True),
		StackUsage(function="a", bytes=4, bounded=True),
		StackUsage(function="b", bytes=8, bounded=True),
		StackUsage(function="c", bytes=16, bounded=True),
		StackUsage(function="d", bytes=2, bounded=True),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="main",
			bound=Unbounded(
				at_least=2 + 4 + 16 + 8 + 2,
				recursion=frozenset({"a", "b", "c", "d"}),
				unmeasured=frozenset(),
				dynamic=frozenset(),
				unresolved=frozenset(),
			),
		),
	)


def test_diamond_dag_depth_accounts_shared_subtrees_once() -> None:
	edges = (
		CallEdge(caller="main", callee="a"),
		CallEdge(caller="main", callee="b"),
		CallEdge(caller="a", callee="c"),
		CallEdge(caller="b", callee="c"),
		CallEdge(caller="c", callee="leaf"),
	)
	frames = (
		StackUsage(function="main", bytes=8, bounded=True),
		StackUsage(function="a", bytes=16, bounded=True),
		StackUsage(function="b", bytes=24, bounded=True),
		StackUsage(function="c", bytes=32, bounded=True),
		StackUsage(function="leaf", bytes=4, bounded=True),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(entry="main", bound=Bounded(bytes=8 + 24 + 32 + 4)),
	)


def test_deep_diamond_completes_with_memoized_subtrees() -> None:
	edges: list[CallEdge] = []
	for index in range(30):
		edges.extend(
			(
				CallEdge(caller=f"layer_{index}", callee=f"side_a_{index}"),
				CallEdge(caller=f"layer_{index}", callee=f"side_b_{index}"),
				CallEdge(caller=f"side_a_{index}", callee=f"layer_{index + 1}"),
				CallEdge(caller=f"side_b_{index}", callee=f"layer_{index + 1}"),
			)
		)
	frames = tuple(
		StackUsage(function=f"layer_{index}", bytes=8, bounded=True) for index in range(31)
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="layer_0",
			bound=Unbounded(
				at_least=31 * 8,
				recursion=frozenset(),
				unmeasured=frozenset(
					f"side_{side}_{index}" for side in ("a", "b") for index in range(30)
				),
				dynamic=frozenset(),
				unresolved=frozenset(),
			),
		),
	)


def test_depth_through_cyclic_nodes_matches_all_paths() -> None:
	edges = (
		CallEdge(caller="main", callee="p"),
		CallEdge(caller="main", callee="q"),
		CallEdge(caller="p", callee="cyc"),
		CallEdge(caller="q", callee="cyc"),
		CallEdge(caller="cyc", callee="cyc"),
		CallEdge(caller="cyc", callee="x1"),
		CallEdge(caller="cyc", callee="x2"),
		CallEdge(caller="x1", callee="leaf"),
		CallEdge(caller="x2", callee="leaf"),
	)
	frames = (
		StackUsage(function="main", bytes=8, bounded=True),
		StackUsage(function="p", bytes=4, bounded=True),
		StackUsage(function="q", bytes=12, bounded=True),
		StackUsage(function="cyc", bytes=16, bounded=True),
		StackUsage(function="x1", bytes=20, bounded=True),
		StackUsage(function="x2", bytes=24, bounded=True),
		StackUsage(function="leaf", bytes=32, bounded=True),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="main",
			bound=Unbounded(
				at_least=8 + 12 + 16 + 24 + 32,
				recursion=frozenset({"cyc"}),
				unmeasured=frozenset(),
				dynamic=frozenset(),
				unresolved=frozenset(),
			),
		),
	)


@pytest.mark.parametrize(("target", "calls"), [(NO_CALLEE, False), (INDIRECT_CALLEE, True)])
def test_a_caller_whose_only_target_can_only_be_0_calls_nothing_in_either_expansion(
	target: str, calls: bool
) -> None:
	targets = {"trampoline": frozenset({"entry"}), "entry": frozenset({target})}
	fallback = frozenset({"cheap"})
	assert (
		any(
			edge.caller == "/src/app.c:entry"
			for edge in expand_indirect_calls(SHARED_ENTRY_EDGES, targets, fallback)
		),
		any(
			edge.caller.endswith(":/src/app.c:entry")
			for edge in own_thread_edges(
				SHARED_ENTRY_EDGES,
				"trampoline",
				"worker_tid",
				"entry",
				ThreadTargets(
					reached=frozenset({"trampoline", "entry"}),
					targets_by_caller=targets,
					sites_by_caller={"trampoline": 1, "entry": 1},
				),
				targets,
				fallback,
			)
		),
	) == (calls, calls)


def test_a_target_only_a_field_narrowing_reaches_reads_indirect_field_in_both_expansions() -> None:
	targets = {"entry": frozenset({"cheap", "costly"})}
	narrowed = {"entry": {"costly": EdgeKind.FIELD}}
	fallback = frozenset({"entry", "cheap", "costly"})
	assert (
		[
			(edge.callee, edge.kind)
			for edge in expand_indirect_calls(
				SHARED_ENTRY_EDGES, targets, fallback, narrowed_by_caller=narrowed
			)
			if edge.caller == "/src/app.c:entry"
		],
		[
			(edge.callee, edge.kind)
			for edge in own_thread_edges(
				SHARED_ENTRY_EDGES,
				"trampoline",
				"worker_tid",
				"entry",
				ThreadTargets(
					reached=frozenset({"trampoline", "entry", "cheap", "costly"}),
					targets_by_caller={
						"trampoline": frozenset({"entry"}),
						"entry": frozenset({"cheap", "costly"}),
					},
					sites_by_caller={"trampoline": 1, "entry": 1},
					narrowed_by_caller=narrowed,
				),
				targets,
				fallback,
				narrowed,
			)
			if edge.caller.endswith(":/src/app.c:entry")
		],
	) == (
		[
			("/src/app.c:entry", EdgeKind.FALLBACK),
			("cheap", EdgeKind.CANDIDATE),
			("costly", EdgeKind.FIELD),
		],
		[
			("worker_tid/:cheap", EdgeKind.CANDIDATE),
			("worker_tid/:costly", EdgeKind.FIELD),
		],
	)


def test_expand_indirect_calls_replaces_placeholders_with_candidates() -> None:
	edges = (
		CallEdge(caller="main", callee="direct_fn"),
		CallEdge(caller="main", callee="__indirect_call"),
		CallEdge(caller="other", callee="__indirect_call"),
	)
	expanded = expand_indirect_calls(
		edges, {"main": frozenset({"cb_b", "cb_a"})}, frozenset({"fallback_fn"})
	)
	assert expanded == (
		CallEdge(caller="main", callee="direct_fn"),
		CallEdge(caller="main", callee="cb_a", kind=EdgeKind.CANDIDATE),
		CallEdge(caller="main", callee="cb_b", kind=EdgeKind.CANDIDATE),
		CallEdge(caller="main", callee="fallback_fn", kind=EdgeKind.FALLBACK),
		CallEdge(caller="other", callee="fallback_fn", kind=EdgeKind.FALLBACK),
	)


def test_expand_indirect_calls_unions_the_fallback_into_caller_sets() -> None:
	edges = (CallEdge(caller="main", callee="__indirect_call"),)
	expanded = expand_indirect_calls(edges, {"main": frozenset({"cb"})}, frozenset({"fb"}))
	assert expanded == (
		CallEdge(caller="main", callee="cb", kind=EdgeKind.CANDIDATE),
		CallEdge(caller="main", callee="fb", kind=EdgeKind.FALLBACK),
	)


@pytest.mark.parametrize("exact", [False, True], ids=["sound", "exact"])
def test_expand_indirect_calls_expands_a_site_without_candidates_to_the_fallback(
	exact: bool,
) -> None:
	edges = (CallEdge(caller="main", callee="__indirect_call"),)
	assert expand_indirect_calls(
		edges, {"main": frozenset({"cb", INDIRECT_CALLEE})}, frozenset({"fb"}), exact=exact
	) == (
		CallEdge(caller="main", callee="cb", kind=EdgeKind.CANDIDATE),
		CallEdge(caller="main", callee="fb", kind=EdgeKind.FALLBACK),
	)


def test_expand_indirect_calls_without_candidates_keeps_placeholders() -> None:
	edges = (CallEdge(caller="main", callee="__indirect_call"),)
	assert expand_indirect_calls(edges, {}, frozenset()) == edges


def test_expand_indirect_calls_matches_path_qualified_callers() -> None:
	edges = (CallEdge(caller="/home/jp/zephyr/shell.c:execute", callee="__indirect_call"),)
	assert expand_indirect_calls(
		edges, {"execute": frozenset({"cb"})}, frozenset({"fallback_fn"})
	) == (
		CallEdge(caller="/home/jp/zephyr/shell.c:execute", callee="cb", kind=EdgeKind.CANDIDATE),
		CallEdge(
			caller="/home/jp/zephyr/shell.c:execute", callee="fallback_fn", kind=EdgeKind.FALLBACK
		),
	)


@pytest.mark.image
@pytest.mark.parametrize("artifacts", ARTIFACT_DIRECTORIES)
def test_every_callgraph_node_meets_its_stack_usage_record(
	zephyr_fixtures: Path, artifacts: str
) -> None:
	usages = load_stack_usages(zephyr_fixtures / artifacts)
	keys = {frame_key(usage.function) for usage in usages}
	roots = {usage.function.split(".")[0] for usage in usages}
	assert (
		sorted(
			name
			for edge in load_callgraph(zephyr_fixtures / artifacts)
			for name in (edge.caller, edge.callee)
			if frame_key(name).split(".")[0] in roots and frame_key(name) not in keys
		)
		== []
	)


_CHORDED_RING = """
from dynamic_call_tree_resolution import CallEdge, StackUsage, worst_case_depths
ring = [f"node_{index}" for index in range(80)]
edges = [CallEdge(caller="main", callee="node_0")]
edges += [CallEdge(caller=node, callee=ring[(index + 1) % 80]) for index, node in enumerate(ring)]
edges += [CallEdge(caller=node, callee=ring[(index * 7) % 80]) for index, node in enumerate(ring)]
frames = [StackUsage(function="main", bytes=8, bounded=True)]
frames += [StackUsage(function=node, bytes=4 + (index * 5) % 17, bounded=True) for index, node in enumerate(ring)]
print([(report.entry, report.bound.at_least) for report in worst_case_depths(edges, frames)])
"""


def test_an_oversized_cycle_bound_does_not_depend_on_the_hash_seed() -> None:
	assert (
		len(
			{
				subprocess.run(
					[sys.executable, "-c", _CHORDED_RING],
					env={**os.environ, "PYTHONHASHSEED": seed},
					capture_output=True,
					text=True,
					check=True,
				).stdout
				for seed in ("1", "2", "3", "4")
			}
		)
		== 1
	)


def _steps(
	edges: tuple[CallEdge, ...], frames: tuple[StackUsage, ...], entry: str
) -> list[tuple[str, int, int]]:
	return [
		(step.function, step.frame, step.cumulative)
		for step in deepest_path(stack_graph(edges, frames), entry)
	]


def test_deepest_path_follows_the_deepest_branch() -> None:
	edges = (
		CallEdge(caller="main", callee="a"),
		CallEdge(caller="main", callee="b"),
		CallEdge(caller="a", callee="leaf"),
		CallEdge(caller="b", callee="c"),
		CallEdge(caller="c", callee="leaf"),
	)
	assert deepest_path(stack_graph(edges, FRAMES), "main") == (
		PathStep(function="main", frame=8, cumulative=8, edge=frozenset(), flags=frozenset()),
		PathStep(
			function="b",
			frame=24,
			cumulative=32,
			edge=frozenset({EdgeKind.STATIC}),
			flags=frozenset(),
		),
		PathStep(
			function="c",
			frame=32,
			cumulative=64,
			edge=frozenset({EdgeKind.STATIC}),
			flags=frozenset(),
		),
		PathStep(
			function="leaf",
			frame=4,
			cumulative=68,
			edge=frozenset({EdgeKind.STATIC}),
			flags=frozenset(),
		),
	)


def test_deepest_path_breaks_a_tie_by_name() -> None:
	edges = (CallEdge(caller="main", callee="b"), CallEdge(caller="main", callee="a"))
	frames = (
		StackUsage(function="main", bytes=8, bounded=True),
		StackUsage(function="a", bytes=4, bounded=True),
		StackUsage(function="b", bytes=4, bounded=True),
	)
	assert _steps(edges, frames, "main") == [("main", 8, 8), ("a", 4, 12)]


def test_deepest_path_through_a_cycle_visits_each_function_once() -> None:
	edges = (
		CallEdge(caller="main", callee="a"),
		CallEdge(caller="a", callee="b"),
		CallEdge(caller="b", callee="a"),
		CallEdge(caller="b", callee="leaf"),
	)
	path = deepest_path(stack_graph(edges, FRAMES), "main")
	assert [(step.function, step.cumulative, step.flags) for step in path] == [
		("main", 8, frozenset()),
		("a", 24, frozenset({Reason.RECURSION})),
		("b", 48, frozenset({Reason.RECURSION})),
		("leaf", 52, frozenset()),
	]


def test_deepest_path_through_an_oversized_cycle_ends_at_the_reported_depth() -> None:
	ring = tuple(f"node_{index}" for index in range(80))
	edges = (
		CallEdge(caller="main", callee="node_0"),
		*(CallEdge(caller=node, callee=ring[(index + 1) % 80]) for index, node in enumerate(ring)),
		*(CallEdge(caller=node, callee=ring[(index * 7) % 80]) for index, node in enumerate(ring)),
		*(CallEdge(caller=node, callee="leaf") for node in ring),
	)
	frames = (
		StackUsage(function="main", bytes=8, bounded=True),
		StackUsage(function="leaf", bytes=100, bounded=True),
		*(
			StackUsage(function=node, bytes=4 + (index * 5) % 17, bounded=True)
			for index, node in enumerate(ring)
		),
	)
	graph = stack_graph(edges, frames)
	path = deepest_path(graph, "main")
	(report,) = stack_reports(graph)
	assert isinstance(report.bound, Unbounded)
	assert path[-1].cumulative == report.bound.at_least
	assert len({step.function for step in path}) == len(path)
	assert path[-1].function == "leaf"


def test_deepest_path_does_not_step_into_the_indirect_placeholder() -> None:
	edges = (
		CallEdge(caller="main", callee="__indirect_call"),
		CallEdge(caller="main", callee="leaf"),
	)
	frames = (
		StackUsage(function="main", bytes=8, bounded=True),
		StackUsage(function="leaf", bytes=0, bounded=True),
	)
	path = deepest_path(stack_graph(edges, frames), "main")
	assert [(step.function, step.flags) for step in path] == [
		("main", frozenset({Reason.UNRESOLVED})),
		("leaf", frozenset()),
	]


@pytest.mark.parametrize(
	("candidate_bytes", "step"),
	[
		pytest.param(40, ("cb", EdgeKind.CANDIDATE), id="through-a-candidate"),
		pytest.param(10, ("fb", EdgeKind.FALLBACK), id="through-the-fallback"),
	],
)
def test_deepest_path_names_the_edge_each_step_is_called_through(
	candidate_bytes: int, step: tuple[str, EdgeKind]
) -> None:
	edges = expand_indirect_calls(
		(CallEdge(caller="main", callee="__indirect_call"),),
		{"main": frozenset({"cb"})},
		frozenset({"fb"}),
	)
	frames = (
		StackUsage(function="main", bytes=8, bounded=True),
		StackUsage(function="cb", bytes=candidate_bytes, bounded=True),
		StackUsage(function="fb", bytes=20, bounded=True),
	)
	(_, callee) = deepest_path(stack_graph(edges, frames), "main")
	assert (callee.function, *callee.edge) == step


def test_deepest_path_of_a_function_outside_the_call_graph_is_its_own_frame() -> None:
	edges = (CallEdge(caller="main", callee="leaf"),)
	frames = (*FRAMES, StackUsage(function="lonely", bytes=12, bounded=False))
	assert deepest_path(stack_graph(edges, frames), "lonely") == (
		PathStep(
			function="lonely",
			frame=12,
			cumulative=12,
			edge=frozenset(),
			flags=frozenset({Reason.DYNAMIC}),
		),
	)


def test_deepest_path_of_a_clone_named_entry_starts_from_the_deeper_clone() -> None:
	edges = (CallEdge(caller="f", callee="a"), CallEdge(caller="f.constprop.0", callee="c"))
	assert _steps(edges, (*FRAMES, StackUsage(function="f", bytes=2, bounded=True)), "f") == [
		("f", 2, 2),
		("c", 32, 34),
	]


def test_deepest_path_of_a_clone_named_entry_starts_from_the_clone_listed_first() -> None:
	edges = (
		CallEdge(caller="f", callee="a"),
		CallEdge(caller="a", callee="a"),
		CallEdge(caller="f.constprop.0", callee="c"),
	)
	graph = stack_graph(edges, (*FRAMES, StackUsage(function="f", bytes=2, bounded=True)))
	first = stack_reports(graph)[0]
	assert isinstance(first.bound, Unbounded)
	assert deepest_path(graph, "f")[-1].cumulative == first.bound.at_least == 18


def test_deepest_path_of_an_unknown_entry_is_an_error() -> None:
	with pytest.raises(ValueError, match="nope is not an entry"):
		deepest_path(stack_graph((CallEdge(caller="main", callee="leaf"),), FRAMES), "nope")


@pytest.fixture(scope="module")
def counter_candidates(
	zephyr_fixtures: Path,
) -> tuple[Mapping[str, frozenset[str]], frozenset[str]]:
	program = load(zephyr_fixtures / ARTIFACT_DIRECTORIES[0] / "zephyr" / "zephyr.exe")
	resolution = resolve(program)
	return per_caller_candidates(program, resolution.sites, resolution.assignments)


def _depth(report: StackReport) -> int:
	return report.bound.bytes if isinstance(report.bound, Bounded) else report.bound.at_least


@pytest.mark.image
def test_no_acyclic_counter_entry_is_deeper_with_candidates_alone_than_with_the_fallback(
	zephyr_fixtures: Path,
	counter_candidates: tuple[Mapping[str, frozenset[str]], frozenset[str]],
) -> None:
	artifacts = zephyr_fixtures / ARTIFACT_DIRECTORIES[0]
	edges = load_callgraph(artifacts)
	frames = load_stack_usages(artifacts)
	candidates, with_fallback = (
		{
			report.entry: report
			for report in worst_case_depths(
				expand_indirect_calls(edges, *counter_candidates, exact=exact),
				frames,
				entry_edges=edges,
			)
		}
		for exact in (True, False)
	)
	assert [
		entry
		for entry, report in with_fallback.items()
		if not (isinstance(report.bound, Unbounded) and report.bound.recursion)
		and _depth(candidates[entry]) > _depth(report)
	] == []


@pytest.mark.image
@pytest.mark.parametrize("expanded", [False, True], ids=["static", "expanded"])
def test_every_counter_entry_has_a_deepest_path_as_deep_as_its_bound(
	zephyr_fixtures: Path,
	counter_candidates: tuple[Mapping[str, frozenset[str]], frozenset[str]],
	expanded: bool,
) -> None:
	artifacts = zephyr_fixtures / ARTIFACT_DIRECTORIES[0]
	edges = load_callgraph(artifacts)
	graph = stack_graph(
		expand_indirect_calls(edges, *counter_candidates) if expanded else edges,
		load_stack_usages(artifacts),
		entry_edges=edges,
	)
	assert [
		report.entry
		for report in stack_reports(graph)
		if (path := deepest_path(graph, report.entry))[-1].cumulative
		!= (report.bound.bytes if isinstance(report.bound, Bounded) else report.bound.at_least)
		or any(
			later.cumulative != earlier.cumulative + later.frame or not later.edge
			for earlier, later in pairwise(path)
		)
	] == []
