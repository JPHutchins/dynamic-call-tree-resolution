# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Tests for :mod:`dynamic_call_tree_resolution.report`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec

from dynamic_call_tree_resolution import (
	AnalysisReport,
	ComparisonReport,
	assignments,
	build_comparison,
	build_report,
	extract_call_sites,
	load,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_report_counts_and_json_round_trip(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["nopie"])
	report = build_report(program, assignments(program), extract_call_sites(program))
	assert report.resolved_slots == 19
	assert report.resolved_targets == 19
	assert {assignment.member_path for assignment in report.assignments} == {
		"ops_a.open",
		"ops_a.close",
		"ops_b.open",
		"ops_b.close",
		"dev_a.api.open",
		"dev_a.api.close",
		"dev_b.api.open",
		"dev_b.api.close",
		"dev_c.api.open",
		"dev_c.api.close",
		"dev_c.context.open",
		"dev_c.context.close",
		"dev_a.ops.init",
		"dev_b.ops.init",
		"dev_c.ops.init",
		"holder2.inner.fn",
		"holder.run",
		"node_a.fn",
		"plain_cb",
	}
	assert any(site.caller == "main" and site.candidates for site in report.call_sites)
	assert {slot.member_path for slot in report.unresolved_slots} == {"bss_cb", "bss_holder.run"}
	assert report.total_slots == 13  # 11 distinct resolved slots + 2 unresolved
	assert msgspec.json.decode(msgspec.json.encode(report), type=AnalysisReport) == report


def test_comparison_rollup_and_json_round_trip(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["nopie"])
	comparison = build_comparison("nopie.elf", program)
	assert comparison == ComparisonReport(
		elf="nopie.elf",
		machine="EM_X86_64",
		functions=21,
		total_slots=13,
		resolved_slots=19,
		unresolved_slots=2,
		call_sites=7,
		resolved_call_sites=5,
		exact_call_sites=5,
		candidate_size_counts=((0, 2), (1, 5)),
	)
	assert msgspec.json.decode(msgspec.json.encode(comparison), type=ComparisonReport) == comparison
