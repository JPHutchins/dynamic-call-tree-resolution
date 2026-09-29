# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Tests for :mod:`dynamic_call_tree_resolution.stack_analysis`."""

from pathlib import Path

import pytest

from dynamic_call_tree_resolution import (
	Bounded,
	CallEdge,
	StackReport,
	StackUsage,
	Unbounded,
	expand_indirect_calls,
	load_callgraph,
	load_stack_usages,
	worst_case_depths,
)
from dynamic_call_tree_resolution.stack_analysis import frame_key

ARTIFACT_DIRECTORIES = (
	Path(__file__).parent / "fixtures" / "counter-su",
	Path(__file__).parent / "fixtures" / "sensor-two-impl",
)

FRAMES = (
	StackUsage(function="main", bytes=8, bounded=True),
	StackUsage(function="a", bytes=16, bounded=True),
	StackUsage(function="b", bytes=24, bounded=True),
	StackUsage(function="c", bytes=32, bounded=True),
	StackUsage(function="leaf", bytes=4, bounded=True),
)


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


def test_oversized_cycles_fail_loudly() -> None:
	edges = tuple(CallEdge(caller=f"node_{i}", callee=f"node_{(i + 1) % 65}") for i in range(65))
	with pytest.raises(ValueError, match="cycle"):
		worst_case_depths(edges, ())


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
		CallEdge(caller="main", callee="cb_a"),
		CallEdge(caller="main", callee="cb_b"),
		CallEdge(caller="main", callee="fallback_fn"),
		CallEdge(caller="other", callee="fallback_fn"),
	)


def test_expand_indirect_calls_unions_the_fallback_into_caller_sets() -> None:
	edges = (CallEdge(caller="main", callee="__indirect_call"),)
	expanded = expand_indirect_calls(edges, {"main": frozenset({"cb"})}, frozenset({"fb"}))
	assert expanded == (
		CallEdge(caller="main", callee="cb"),
		CallEdge(caller="main", callee="fb"),
	)


def test_expand_indirect_calls_without_candidates_keeps_placeholders() -> None:
	edges = (CallEdge(caller="main", callee="__indirect_call"),)
	assert expand_indirect_calls(edges, {}, frozenset()) == edges


def test_expand_indirect_calls_matches_path_qualified_callers() -> None:
	edges = (CallEdge(caller="/home/jp/zephyr/shell.c:execute", callee="__indirect_call"),)
	assert expand_indirect_calls(
		edges, {"execute": frozenset({"cb"})}, frozenset({"fallback_fn"})
	) == (
		CallEdge(caller="/home/jp/zephyr/shell.c:execute", callee="cb"),
		CallEdge(caller="/home/jp/zephyr/shell.c:execute", callee="fallback_fn"),
	)


@pytest.mark.parametrize(
	"artifacts",
	ARTIFACT_DIRECTORIES,
	ids=[artifacts.name for artifacts in ARTIFACT_DIRECTORIES],
)
def test_every_callgraph_node_meets_its_stack_usage_record(artifacts: Path) -> None:
	usages = load_stack_usages(artifacts)
	keys = {frame_key(usage.function) for usage in usages}
	roots = {usage.function.split(".")[0] for usage in usages}
	assert (
		sorted(
			name
			for edge in load_callgraph(artifacts)
			for name in (edge.caller, edge.callee)
			if frame_key(name).split(".")[0] in roots and frame_key(name) not in keys
		)
		== []
	)
