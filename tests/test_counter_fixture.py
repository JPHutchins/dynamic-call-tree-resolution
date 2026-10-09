# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The counter build artifacts pin the summary and the README's claims."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import pytest

from dynamic_call_tree_resolution import AnalysisSummary, UnboundedStack, load
from dynamic_call_tree_resolution.cli import stack
from dynamic_call_tree_resolution.report import BoundedStack, StackEntryReport
from tests.dctr import dctr

if TYPE_CHECKING:
	from pathlib import Path

pytestmark = pytest.mark.image


@pytest.fixture(scope="module")
def artifacts(zephyr_fixtures: Path) -> Path:
	return zephyr_fixtures / "counter-su"


@pytest.fixture(scope="module")
def executable(artifacts: Path) -> Path:
	return artifacts / "zephyr" / "zephyr.exe"


def test_summary_pins_the_published_numbers(artifacts: Path, executable: Path) -> None:
	summary = msgspec.json.decode(
		dctr("summary", str(artifacts), str(executable)), type=AnalysisSummary
	)
	assert summary == AnalysisSummary(
		resolved_slots=106,
		total_slots=250,
		unresolved_slots=144,
		resolved_targets=200,
		indirect_call_sites=83,
		total_functions=729,
		entry_points=157,
		discarded_entry_points=286,
		worst_case_entry="cmd_prompt_off",
		worst_case=summary.worst_case,
		rtos="zephyr",
	)
	assert isinstance(summary.worst_case, UnboundedStack)
	assert (
		summary.worst_case.at_least_bytes,
		len(summary.worst_case.recursion),
		len(summary.worst_case.unmeasured),
		summary.worst_case.dynamic,
		summary.worst_case.unresolved,
	) == (6160, 234, 149, (), ())


def test_shell_readline_is_not_in_the_linked_executable(executable: Path) -> None:
	assert "shell_readline" not in {
		function.name for function in load(executable).functions.values()
	}


@pytest.mark.parametrize("elf", [True, False], ids=["with the image", "from the artifacts"])
def test_an_assumed_frame_bounds_the_rows_it_alone_left_unmeasured(
	artifacts: Path, executable: Path, capsys: pytest.CaptureFixture[str], elf: bool
) -> None:
	stack(
		artifacts,
		executable if elf else None,
		assume_frame=("nsi_vprint_error_and_exit=48",),
		json=True,
	)
	assert {
		row.entry: row.bound
		for row in msgspec.json.decode(capsys.readouterr().out, type=tuple[StackEntryReport, ...])
		if row.entry == "arch_system_halt"
	} == {
		"arch_system_halt": BoundedStack(
			bytes=32 + 32 + 48, assumed_frames=("nsi_vprint_error_and_exit",)
		)
	}


@pytest.mark.parametrize(
	("statement", "message"),
	[
		pytest.param("nsi_vprint_error_and_exit", "is not FUNCTION=BYTES", id="no size"),
		pytest.param("nsi_vprint_error_and_exit=big", "is not FUNCTION=BYTES", id="no number"),
		pytest.param("arch_system_halt=8", "already has a frame", id="a measured function"),
		pytest.param("no_such_function=8", "no function named", id="an unknown function"),
	],
)
def test_assume_frame_states_a_size_for_a_function_without_one(
	artifacts: Path, executable: Path, statement: str, message: str
) -> None:
	with pytest.raises(ValueError, match=message):
		stack(artifacts, executable, assume_frame=(statement,))
