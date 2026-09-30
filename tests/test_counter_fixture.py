# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The counter build artifacts pin the summary and the README's claims."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import msgspec
import pytest

from dynamic_call_tree_resolution import AnalysisSummary, UnboundedStack, load

pytestmark = pytest.mark.image


@pytest.fixture(scope="module")
def artifacts(zephyr_fixtures: Path) -> Path:
	return zephyr_fixtures / "counter-su"


@pytest.fixture(scope="module")
def executable(artifacts: Path) -> Path:
	return artifacts / "zephyr" / "zephyr.exe"


def _dctr(*arguments: str) -> str:
	executable = Path(sys.executable).with_name("dctr")
	result = subprocess.run(
		[str(executable), *arguments],
		check=True,
		capture_output=True,
		text=True,
	)
	return result.stdout


def test_summary_pins_the_published_numbers(artifacts: Path, executable: Path) -> None:
	summary = msgspec.json.decode(
		_dctr("summary", str(artifacts), str(executable)), type=AnalysisSummary
	)
	assert summary == AnalysisSummary(
		resolved_slots=106,
		total_slots=250,
		unresolved_slots=144,
		resolved_targets=200,
		indirect_call_sites=101,
		total_functions=729,
		entry_points=148,
		discarded_entry_points=286,
		worst_case_entry="gpio_emul_port_set_masked_raw",
		worst_case=summary.worst_case,
	)
	assert isinstance(summary.worst_case, UnboundedStack)
	assert (
		summary.worst_case.at_least_bytes,
		len(summary.worst_case.recursion),
		len(summary.worst_case.unmeasured),
		summary.worst_case.dynamic,
		summary.worst_case.unresolved,
	) == (6288, 222, 153, (), ())


def test_shell_readline_is_not_in_the_linked_executable(executable: Path) -> None:
	assert "shell_readline" not in {
		function.name for function in load(executable).functions.values()
	}
