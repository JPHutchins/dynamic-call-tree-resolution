# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Tests for :mod:`dynamic_call_tree_resolution.stack_analysis`."""

from dynamic_call_tree_resolution import (
	CallEdge,
	StackReport,
	StackUsage,
	expand_indirect_calls,
	worst_case_depths,
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
		StackReport(entry="main", depth=8 + 24 + 32 + 4, recursive=False, has_dynamic=False),
	)


def test_worst_case_depth_flags_recursion() -> None:
	edges = (
		CallEdge(caller="main", callee="a"),
		CallEdge(caller="a", callee="a"),
		CallEdge(caller="a", callee="leaf"),
	)
	report = worst_case_depths(edges, FRAMES)
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
		StackReport(entry="main", depth=12, recursive=False, has_dynamic=True),
	)


def test_worst_case_depth_counts_missing_frames_as_zero() -> None:
	edges = (CallEdge(caller="main", callee="no_record"),)
	assert worst_case_depths(edges, FRAMES) == (
		StackReport(entry="main", depth=8, recursive=False, has_dynamic=False),
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
		StackReport(entry="main", depth=8 + 64 + 4, recursive=False, has_dynamic=True),
	)


def test_clone_suffixes_match_across_sources() -> None:
	edges = (CallEdge(caller="main", callee="k_sleep_ticks.isra.0"),)
	frames = (
		StackUsage(function="main", bytes=8, dynamic=False),
		StackUsage(function="k_sleep_ticks.isra", bytes=32, dynamic=False),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(entry="main", depth=8 + 32, recursive=False, has_dynamic=False),
	)


def test_duplicate_su_names_keep_the_largest_frame() -> None:
	edges = (CallEdge(caller="bg_thread_main", callee="main"),)
	frames = (
		StackUsage(function="bg_thread_main", bytes=8, dynamic=False),
		StackUsage(function="main", bytes=4, dynamic=False),
		StackUsage(function="main", bytes=160, dynamic=False),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(entry="bg_thread_main", depth=8 + 160, recursive=False, has_dynamic=False),
	)


def test_duplicate_su_names_union_the_dynamic_flag() -> None:
	edges = (CallEdge(caller="bg_thread_main", callee="main"),)
	frames = (
		StackUsage(function="bg_thread_main", bytes=8, dynamic=False),
		StackUsage(function="main", bytes=160, dynamic=False),
		StackUsage(function="main", bytes=4, dynamic=True),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(entry="bg_thread_main", depth=168, recursive=False, has_dynamic=True),
	)


def test_static_entry_points_report_bare_names() -> None:
	edges = (CallEdge(caller="/home/jp/zephyr/main.c:static_entry", callee="leaf"),)
	frames = (
		StackUsage(function="static_entry", bytes=16, dynamic=False),
		StackUsage(function="leaf", bytes=4, dynamic=False),
	)
	assert worst_case_depths(edges, frames) == (
		StackReport(entry="static_entry", depth=20, recursive=False, has_dynamic=False),
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
		CallEdge(caller="other", callee="fallback_fn"),
	)


def test_expand_indirect_calls_without_candidates_drops_placeholders() -> None:
	edges = (CallEdge(caller="main", callee="__indirect_call"),)
	assert expand_indirect_calls(edges, {}, frozenset()) == ()


def test_expand_indirect_calls_matches_path_qualified_callers() -> None:
	edges = (CallEdge(caller="/home/jp/zephyr/shell.c:execute", callee="__indirect_call"),)
	assert expand_indirect_calls(
		edges, {"execute": frozenset({"cb"})}, frozenset({"fallback_fn"})
	) == (CallEdge(caller="/home/jp/zephyr/shell.c:execute", callee="cb"),)
