# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Tests for :mod:`dynamic_call_tree_resolution.report`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dynamic_call_tree_resolution import AnalysisReport, assignments, build_report, load

if TYPE_CHECKING:
	from pathlib import Path


def test_report_counts_and_json_round_trip(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["nopie"])
	report = build_report(program, assignments(program))
	assert report.resolved_slots == 11
	assert report.resolved_targets == 11
	assert {assignment.member_path for assignment in report.assignments} == {
		"ops_a.open",
		"ops_a.close",
		"ops_b.open",
		"ops_b.close",
		"dev_a.api.open",
		"dev_a.api.close",
		"dev_b.api.open",
		"dev_b.api.close",
		"holder.run",
		"node_a.fn",
		"plain_cb",
	}
	assert AnalysisReport.model_validate_json(report.model_dump_json()) == report
