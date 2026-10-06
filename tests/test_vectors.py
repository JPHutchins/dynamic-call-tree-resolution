# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The handlers an M-profile vector table holds, and those only the hardware can call."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import pytest
from salix import replace

from dynamic_call_tree_resolution import Address, load
from dynamic_call_tree_resolution.cli import stack
from dynamic_call_tree_resolution.report import BoundedStack, StackEntryReport, UnboundedStack
from dynamic_call_tree_resolution.vsa.vectors import hardware_handlers, vector_table
from tests.toolchains import FIXTURES, build_cortex_m3

if TYPE_CHECKING:
	from pathlib import Path


@pytest.mark.parametrize(
	("source", "handlers"),
	[
		("minimal.c", ["fault", "reset"]),
		("reproducers/kept_handler.c", ["fault"]),
	],
)
def test_a_handler_is_hardware_only_when_the_vector_table_alone_holds_its_address(
	tmp_path: Path, source: str, handlers: list[str]
) -> None:
	program = load(build_cortex_m3((FIXTURES / source,), tmp_path / "image.elf"))
	assert (
		sorted(program.functions[handler].name for handler in vector_table(program).values()),
		sorted(program.functions[handler].name for handler in hardware_handlers(program)),
	) == (["fault", "fault", "reset"], handlers)


@pytest.mark.image
def test_sensor_threads_vector_table_holds_only_handlers_the_software_never_stores(
	zephyr_fixtures: Path,
) -> None:
	program = load(zephyr_fixtures / "sensor-threads" / "zephyr" / "zephyr.elf")
	assert (
		len(vector_table(program)),
		sorted(program.functions[handler].name for handler in hardware_handlers(program)),
	) == (
		53,
		[
			"__start",
			"_isr_wrapper",
			"sys_clock_isr",
			"z_arm_hard_fault",
			"z_arm_nmi",
			"z_arm_pendsv",
			"z_arm_svc",
		],
	)


def test_an_image_without_an_entry_point_has_no_vector_table(tmp_path: Path) -> None:
	program = load(build_cortex_m3((FIXTURES / "minimal.c",), tmp_path / "image.elf"))
	assert vector_table(replace(program, entry_point=Address(0))) == {}


@pytest.mark.image
def test_each_handler_only_the_hardware_calls_is_a_row_of_its_own(
	zephyr_fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	artifacts = zephyr_fixtures / "sensor-threads"
	stack(artifacts, artifacts / "zephyr" / "zephyr.elf", json=True)
	rows = {
		row.entry: row.bound
		for row in msgspec.json.decode(capsys.readouterr().out, type=tuple[StackEntryReport, ...])
	}
	assert (
		{"_isr_wrapper", "sys_clock_isr"} <= rows.keys(),
		{
			entry: rows[entry]
			for entry in ("__start", "z_arm_hard_fault", "z_arm_nmi", "z_arm_pendsv", "z_arm_svc")
		},
	) == (
		True,
		{
			"__start": _bare("__start"),
			"z_arm_hard_fault": _bare("z_arm_hard_fault"),
			"z_arm_nmi": BoundedStack(bytes=8, measured=("z_SysNmiOnReset",)),
			"z_arm_pendsv": BoundedStack(bytes=0, measured=("z_arm_pendsv",)),
			"z_arm_svc": _bare("z_arm_svc"),
		},
	)


def _bare(handler: str) -> UnboundedStack:
	return UnboundedStack(
		at_least_bytes=0, recursion=(), unmeasured=(handler,), dynamic=(), unresolved=()
	)
