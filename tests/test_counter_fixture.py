# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The counter build artifacts pin the README's published numbers.

``tests/fixtures/counter-su`` holds the Zephyr CAN counter build's linked
executable and its ``-fstack-usage``/``-fcallgraph-info`` artifacts
(produced by the ``counter_su`` task); the README's comparison numbers
are the outputs asserted here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import msgspec

from dynamic_call_tree_resolution import AnalysisSummary

ARTIFACTS = Path(__file__).parent / "fixtures" / "counter-su"
EXECUTABLE = ARTIFACTS / "zephyr" / "zephyr.exe"


def _dctr(*arguments: str) -> str:
	executable = Path(sys.executable).with_name("dctr")
	result = subprocess.run(
		[str(executable), *arguments],
		check=True,
		capture_output=True,
		text=True,
	)
	return result.stdout


def test_summary_pins_the_published_numbers() -> None:
	summary = msgspec.json.decode(
		_dctr("summary", str(ARTIFACTS), str(EXECUTABLE)), type=AnalysisSummary
	)
	assert summary == AnalysisSummary(
		resolved_slots=70,
		total_slots=121,
		unresolved_slots=51,
		resolved_targets=145,
		indirect_call_sites=100,
		total_functions=734,
		entry_points=287,
		worst_case_bytes=2108,
		worst_case_entry="shell_readline",
	)


def test_stack_pins_the_published_depths() -> None:
	output = _dctr("stack", str(ARTIFACTS))
	assert "poll_state_thread: 460 bytes" in output
	assert "shell_readline: 1760 bytes recursive" in output


def test_compare_pins_the_published_rollup() -> None:
	output = _dctr("compare", str(EXECUTABLE))
	assert "70/51/121" in output
	assert "6/6/98" in output
