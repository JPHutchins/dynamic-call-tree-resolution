# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The sensor-two-impl fixture pins the per-thread resolution proof.

``tests/fixtures/sensor-two-impl-app`` builds two identical threads,
each bound to a different existing sensor driver (ADT7420 and BMI160)
on one emulated I2C bus; the committed artifacts (produced by the
``sensor_two_impl`` task) let the tests here replay the analysis and
pin each thread's dispatch to its own impl.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import msgspec

from dynamic_call_tree_resolution import (
	AnalysisReport,
	assignments,
	extract_call_sites,
	load,
)
from dynamic_call_tree_resolution.call_sites import per_caller_candidates
from dynamic_call_tree_resolution.callgraph import load_callgraph
from dynamic_call_tree_resolution.stack_analysis import (
	expand_indirect_calls,
	worst_case_depths,
)
from dynamic_call_tree_resolution.stack_usage import load_stack_usages

ARTIFACTS = Path(__file__).parent / "fixtures" / "sensor-two-impl"
EXECUTABLE = ARTIFACTS / "zephyr" / "zephyr.elf"


def _dctr(*arguments: str) -> str:
	executable = Path(sys.executable).with_name("dctr")
	result = subprocess.run(
		[str(executable), *arguments],
		check=True,
		capture_output=True,
		text=True,
	)
	return result.stdout


def test_analyze_resolves_each_threads_dispatch_to_its_own_impl() -> None:
	report = msgspec.json.decode(_dctr("analyze", "--json", str(EXECUTABLE)), type=AnalysisReport)
	sites_by_caller: dict[str, set[str]] = {}
	for site in report.call_sites:
		sites_by_caller.setdefault(site.caller, set()).update(
			candidate.name for candidate in site.candidates
		)
	assert sites_by_caller["thermal_thread"] == {"adt7420_sample_fetch", "adt7420_channel_get"}
	assert sites_by_caller["motion_thread"] == {"bmi160_sample_fetch", "bmi160_channel_get"}


def test_exact_expansion_gives_each_thread_its_impl_depth() -> None:
	program = load(EXECUTABLE)
	edges = load_callgraph(ARTIFACTS)
	targets_by_caller, fallback = per_caller_candidates(
		program, extract_call_sites(program), assignments(program)
	)
	exact = expand_indirect_calls(edges, targets_by_caller, fallback, exact=True)
	reports = worst_case_depths(exact, load_stack_usages(ARTIFACTS), entry_edges=edges)
	depths = {report.entry: report for report in reports}
	assert depths["thermal_thread"].depth == 72
	assert depths["thermal_thread"].recursive is False
	assert depths["motion_thread"].depth == 648
	assert depths["motion_thread"].recursive is True
