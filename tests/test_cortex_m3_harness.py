# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The Cortex-M3 harness runs a fixture program on QEMU, the ground truth for ARM images."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dynamic_call_tree_resolution import (
	Machine,
	assignments,
	build_report,
	extract_call_sites,
	load,
)
from tests.toolchains import CORTEX_M3_HARNESS, build_cortex_m3, run_cortex_m3

if TYPE_CHECKING:
	from pathlib import Path


@pytest.fixture(scope="module")
def harness_check(tmp_path_factory: pytest.TempPathFactory) -> Path:
	return build_cortex_m3(
		(CORTEX_M3_HARNESS / "harness_check.c",),
		tmp_path_factory.mktemp("cortex-m3") / "harness_check.elf",
		"-g",
		"-O2",
	)


def test_harness_passes_arguments_and_returns_mains_status(harness_check: Path) -> None:
	result = run_cortex_m3(harness_check, "x", "y")
	assert result.stdout.splitlines() == ["harness_check.elf", "x", "y", "initialized_target"]
	assert result.returncode == 3


def test_harness_exits_on_a_fault_instead_of_hanging(harness_check: Path) -> None:
	result = run_cortex_m3(harness_check, "x", "y", "z")
	assert result.stdout.splitlines() == ["harness_check.elf", "x", "y", "z", "initialized_target"]
	assert result.returncode == 1


def test_harness_image_analyzes_as_a_thumb_program(harness_check: Path) -> None:
	program = load(harness_check)
	report = build_report(program, assignments(program), extract_call_sites(program))
	assert program.machine == Machine.EM_ARM
	assert {
		assignment.member_path: tuple(candidate.name for candidate in assignment.candidates)
		for assignment in report.assignments
	} == {
		"vectors.reset": ("reset",),
		"vectors.nmi": ("fault",),
		"vectors.hard_fault": ("fault",),
		"initialized": ("initialized_target",),
	}
	assert [slot.member_path for slot in report.unresolved_slots] == ["zeroed"]
