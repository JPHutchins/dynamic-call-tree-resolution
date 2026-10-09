# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Tests for :mod:`dynamic_call_tree_resolution.report`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import pytest

from dynamic_call_tree_resolution import (
	AnalysisReport,
	Bounded,
	ComparisonReport,
	FunctionComparison,
	Machine,
	Unbounded,
	UnboundedStack,
	assignments,
	build_comparison,
	build_report,
	extract_call_sites,
	load,
)
from dynamic_call_tree_resolution.report import stack_bound_report
from tests.expected import EXPECTED_PATHS

if TYPE_CHECKING:
	from pathlib import Path


def test_report_counts_and_json_round_trip(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["nopie"])
	report = build_report(program, assignments(program), extract_call_sites(program))
	assert report.resolved_slots == 11
	assert report.resolved_targets == 19
	assert {assignment.member_path for assignment in report.assignments} == EXPECTED_PATHS
	assert any(site.caller == "main" and site.candidates for site in report.call_sites)
	assert {slot.member_path for slot in report.unresolved_slots} == {"bss_cb", "bss_holder.run"}
	assert report.total_slots == 13  # 11 distinct resolved slots + 2 unresolved
	assert report.resolved_slots + len(report.unresolved_slots) == report.total_slots
	assert msgspec.json.decode(msgspec.json.encode(report), type=AnalysisReport) == report


def test_comparison_rollup_and_json_round_trip(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["nopie"])
	addresses = {function.name: function.address for function in program.functions.values()}
	comparison = build_comparison("nopie.elf", program)
	assert comparison == ComparisonReport(
		elf="nopie.elf",
		machine=Machine.EM_X86_64,
		functions=21,
		total_slots=13,
		resolved_slots=11,
		unresolved_slots=2,
		call_sites=10,
		resolved_call_sites=7,
		exact_call_sites=7,
		candidate_size_counts=((0, 5), (1, 5)),
		pexplorer_dynamic_sites=None,
		function_comparisons=(
			*(
				FunctionComparison(
					address=addresses[caller] & ~1,
					caller=caller,
					pexplorer_dynamic_sites=0,
					dctr_call_sites=1,
					dctr_resolved_sites=0,
					dctr_exact_sites=0,
				)
				for caller in ("_init", "_start")
			),
			FunctionComparison(
				address=addresses["deregister_tm_clones"] & ~1,
				caller="deregister_tm_clones",
				pexplorer_dynamic_sites=0,
				dctr_call_sites=1,
				dctr_resolved_sites=1,
				dctr_exact_sites=1,
			),
			FunctionComparison(
				address=addresses["main"] & ~1,
				caller="main",
				pexplorer_dynamic_sites=0,
				dctr_call_sites=6,
				dctr_resolved_sites=5,
				dctr_exact_sites=5,
			),
			FunctionComparison(
				address=addresses["register_tm_clones"] & ~1,
				caller="register_tm_clones",
				pexplorer_dynamic_sites=0,
				dctr_call_sites=1,
				dctr_resolved_sites=1,
				dctr_exact_sites=1,
			),
		),
	)
	assert msgspec.json.decode(msgspec.json.encode(comparison), type=ComparisonReport) == comparison


def test_a_read_only_null_slot_counts_as_resolved_to_nothing(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["residue"])
	report = build_report(program, assignments(program), extract_call_sites(program))
	assert (
		tuple(slot.member_path for slot in report.null_slots),
		tuple(slot.member_path for slot in report.unresolved_slots),
		report.resolved_slots,
		report.total_slots,
	) == (
		("rom_ops.stop",),
		("rom_arm.run", "ram_ops.stop", "bss_ops.run", "bss_ops.stop"),
		4,
		8,
	)


@pytest.mark.image
def test_a_dead_site_counts_as_resolved_and_exact(zephyr_fixtures: Path) -> None:
	rows = {
		row.caller: row
		for row in build_comparison(
			"hello", load(zephyr_fixtures / "hello" / "zephyr" / "zephyr.elf")
		).function_comparisons
	}
	assert tuple(
		(
			rows[caller].dctr_call_sites,
			rows[caller].dctr_resolved_sites,
			rows[caller].dctr_exact_sites,
		)
		for caller in ("bg_thread_main", "z_cstart")
	) == ((1, 1, 1), (1, 1, 1))


def test_a_bounded_stack_serializes_its_bytes_and_its_measured_frames_sorted() -> None:
	assert (
		msgspec.json.encode(
			stack_bound_report(Bounded(bytes=16, measured=frozenset({"memset", "__aeabi_ldivmod"})))
		)
		== b'{"kind":"bounded","bytes":16,"measured":["__aeabi_ldivmod","memset"],"stack_reservation_bytes":0,"exception_frame_bytes":0,"assumed_no_recursion":[],"narrowed_by_field":[],"narrowed_by_signature":[],"assumed_unwritten":[],"assumed_frames":[]}'
	)


def test_an_unbounded_stack_serializes_its_reasons_sorted() -> None:
	assert stack_bound_report(
		Unbounded(
			at_least=8,
			recursion=frozenset({"b", "a"}),
			unmeasured=frozenset({"asm"}),
			dynamic=frozenset(),
			unresolved=frozenset({"main"}),
		)
	) == UnboundedStack(
		at_least_bytes=8,
		recursion=("a", "b"),
		unmeasured=("asm",),
		dynamic=(),
		unresolved=("main",),
	)
