# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Tests for :mod:`dynamic_call_tree_resolution.stack_analysis`."""

from pathlib import Path

import pytest

from dynamic_call_tree_resolution import (
	CallEdge,
	StackReport,
	StackUsage,
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
	StackUsage(function="main", bytes=8, dynamic=False),
	StackUsage(function="a", bytes=16, dynamic=False),
	StackUsage(function="b", bytes=24, dynamic=False),
	StackUsage(function="c", bytes=32, dynamic=False),
	StackUsage(function="leaf", bytes=4, dynamic=False),
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
		StackReport(
			entry="main", depth=8 + 24 + 32 + 4, recursive=False, has_dynamic=False, unmeasured=0
		),
	)


def test_worst_case_depth_flags_recursion() -> None:
	edges = (
		CallEdge(caller="main", callee="a"),
		CallEdge(caller="a", callee="a"),
		CallEdge(caller="a", callee="leaf"),
	)
	frames = (
		StackUsage(function="main", bytes=8, dynamic=False),
		StackUsage(function="a", bytes=16, dynamic=False),
		StackUsage(function="leaf", bytes=4, dynamic=False),
	)
	report = worst_case_depths(edges, frames)
	assert report[0].entry == "main"
	assert report[0].recursive
	assert report[0].depth == 8 + 16 + 4


def test_worst_case_depth_flags_dynamic_frames() -> None:
	edges = (CallEdge(caller="main", callee="leaf"),)
	frames = (
		StackUsage(function="main", bytes=8, dynamic=False),
		StackUsage(function="leaf", bytes=4, dynamic=True),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(entry="main", depth=12, recursive=False, has_dynamic=True, unmeasured=0),
	)


def test_worst_case_depth_counts_missing_frames_as_unmeasured() -> None:
	edges = (CallEdge(caller="main", callee="no_record"),)
	frames = (StackUsage(function="main", bytes=8, dynamic=False),)
	assert worst_case_depths(edges, frames) == (
		StackReport(entry="main", depth=8, recursive=False, has_dynamic=False, unmeasured=1),
	)


def test_path_qualified_static_names_match_su_records() -> None:
	edges = (
		CallEdge(caller="main", callee="/home/jp/zephyr/kernel/timeslicing.c:slice_reset"),
		CallEdge(caller="/home/jp/zephyr/kernel/timeslicing.c:slice_reset", callee="leaf"),
	)
	frames = (
		StackUsage(function="main", bytes=8, dynamic=False),
		StackUsage(function="slice_reset", bytes=64, dynamic=True),
		StackUsage(function="leaf", bytes=4, dynamic=False),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="main", depth=8 + 64 + 4, recursive=False, has_dynamic=True, unmeasured=0
		),
	)


def test_clone_suffixes_match_across_sources() -> None:
	edges = (CallEdge(caller="main", callee="k_sleep_ticks.isra.0"),)
	frames = (
		StackUsage(function="main", bytes=8, dynamic=False),
		StackUsage(function="k_sleep_ticks.isra", bytes=32, dynamic=False),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(entry="main", depth=8 + 32, recursive=False, has_dynamic=False, unmeasured=0),
	)


def test_duplicate_su_names_keep_the_largest_frame() -> None:
	edges = (CallEdge(caller="bg_thread_main", callee="main"),)
	frames = (
		StackUsage(function="bg_thread_main", bytes=8, dynamic=False),
		StackUsage(function="main", bytes=4, dynamic=False),
		StackUsage(function="main", bytes=160, dynamic=False),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="bg_thread_main", depth=8 + 160, recursive=False, has_dynamic=False, unmeasured=0
		),
	)


def test_duplicate_su_names_union_the_dynamic_flag() -> None:
	edges = (CallEdge(caller="bg_thread_main", callee="main"),)
	frames = (
		StackUsage(function="bg_thread_main", bytes=8, dynamic=False),
		StackUsage(function="main", bytes=160, dynamic=False),
		StackUsage(function="main", bytes=4, dynamic=True),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="bg_thread_main", depth=168, recursive=False, has_dynamic=True, unmeasured=0
		),
	)


def test_static_entry_points_report_bare_names() -> None:
	edges = (CallEdge(caller="/home/jp/zephyr/main.c:static_entry", callee="leaf"),)
	frames = (
		StackUsage(function="static_entry", bytes=16, dynamic=False),
		StackUsage(function="leaf", bytes=4, dynamic=False),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="static_entry", depth=20, recursive=False, has_dynamic=False, unmeasured=0
		),
	)


def test_same_named_statics_in_different_files_stay_distinct_nodes() -> None:
	edges = (
		CallEdge(caller="main", callee="/home/jp/zephyr/a.c:helper"),
		CallEdge(caller="main", callee="/home/jp/zephyr/b.c:helper"),
		CallEdge(caller="/home/jp/zephyr/a.c:helper", callee="leaf_a"),
		CallEdge(caller="/home/jp/zephyr/b.c:helper", callee="leaf_b"),
	)
	frames = (
		StackUsage(function="main", bytes=8, dynamic=False),
		StackUsage(function="helper", bytes=16, dynamic=False),
		StackUsage(function="leaf_a", bytes=40, dynamic=False),
		StackUsage(function="leaf_b", bytes=24, dynamic=False),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="main", depth=8 + 16 + 40, recursive=False, has_dynamic=False, unmeasured=0
		),
	)


def test_flags_propagate_from_non_deepest_branches() -> None:
	edges = (
		CallEdge(caller="main", callee="deep"),
		CallEdge(caller="main", callee="shallow"),
		CallEdge(caller="deep", callee="leaf"),
	)
	frames = (
		StackUsage(function="main", bytes=8, dynamic=False),
		StackUsage(function="deep", bytes=40, dynamic=False),
		StackUsage(function="shallow", bytes=4, dynamic=True),
		StackUsage(function="leaf", bytes=4, dynamic=False),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="main", depth=8 + 40 + 4, recursive=False, has_dynamic=True, unmeasured=0
		),
	)


def test_recursion_flag_propagates_from_non_deepest_branches() -> None:
	edges = (
		CallEdge(caller="main", callee="deep"),
		CallEdge(caller="main", callee="shallow"),
		CallEdge(caller="deep", callee="leaf"),
		CallEdge(caller="shallow", callee="shallow"),
	)
	frames = (
		StackUsage(function="main", bytes=8, dynamic=False),
		StackUsage(function="deep", bytes=40, dynamic=False),
		StackUsage(function="shallow", bytes=4, dynamic=False),
		StackUsage(function="leaf", bytes=4, dynamic=False),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="main", depth=8 + 40 + 4, recursive=True, has_dynamic=False, unmeasured=0
		),
	)


def test_edgeless_su_functions_are_entry_points() -> None:
	edges = (CallEdge(caller="main", callee="leaf"),)
	frames = (
		StackUsage(function="main", bytes=8, dynamic=False),
		StackUsage(function="leaf", bytes=4, dynamic=False),
		StackUsage(function="isr", bytes=64, dynamic=False),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(entry="isr", depth=64, recursive=False, has_dynamic=False, unmeasured=0),
		StackReport(entry="main", depth=12, recursive=False, has_dynamic=False, unmeasured=0),
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
		StackUsage(function="main", bytes=2, dynamic=False),
		StackUsage(function="a", bytes=4, dynamic=False),
		StackUsage(function="b", bytes=8, dynamic=False),
		StackUsage(function="c", bytes=16, dynamic=False),
		StackUsage(function="d", bytes=2, dynamic=False),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="main", depth=2 + 4 + 16 + 8 + 2, recursive=True, has_dynamic=False, unmeasured=0
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
		StackUsage(function="main", bytes=8, dynamic=False),
		StackUsage(function="a", bytes=16, dynamic=False),
		StackUsage(function="b", bytes=24, dynamic=False),
		StackUsage(function="c", bytes=32, dynamic=False),
		StackUsage(function="leaf", bytes=4, dynamic=False),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="main", depth=8 + 24 + 32 + 4, recursive=False, has_dynamic=False, unmeasured=0
		),
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
		StackUsage(function=f"layer_{index}", bytes=8, dynamic=False) for index in range(31)
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="layer_0", depth=31 * 8, recursive=False, has_dynamic=False, unmeasured=30
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
		StackUsage(function="main", bytes=8, dynamic=False),
		StackUsage(function="p", bytes=4, dynamic=False),
		StackUsage(function="q", bytes=12, dynamic=False),
		StackUsage(function="cyc", bytes=16, dynamic=False),
		StackUsage(function="x1", bytes=20, dynamic=False),
		StackUsage(function="x2", bytes=24, dynamic=False),
		StackUsage(function="leaf", bytes=32, dynamic=False),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(
			entry="main",
			depth=8 + 12 + 16 + 24 + 32,
			recursive=True,
			has_dynamic=False,
			unmeasured=0,
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


def test_expand_indirect_calls_without_candidates_drops_placeholders() -> None:
	edges = (CallEdge(caller="main", callee="__indirect_call"),)
	assert expand_indirect_calls(edges, {}, frozenset()) == ()


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
