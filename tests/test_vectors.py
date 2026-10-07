# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The handlers an M-profile vector table holds, and those only the hardware can call."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

import msgspec
import pytest
from salix import replace

from dynamic_call_tree_resolution import Address, load
from dynamic_call_tree_resolution.cli import stack
from dynamic_call_tree_resolution.report import BoundedStack, StackEntryReport, UnboundedStack
from dynamic_call_tree_resolution.stack_analysis import LevelSource
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
			entry: _shown(rows[entry])
			for entry in ("__start", "z_arm_hard_fault", "z_arm_nmi", "z_arm_pendsv", "z_arm_svc")
		},
	) == (
		True,
		{
			"__start": (1016, ()),
			"z_arm_hard_fault": (1048, ()),
			"z_arm_nmi": BoundedStack(bytes=8, measured=("z_SysNmiOnReset",)),
			"z_arm_pendsv": BoundedStack(bytes=0, measured=("z_arm_pendsv",)),
			"z_arm_svc": (944, ()),
		},
	)


def _shown(bound: BoundedStack | UnboundedStack) -> BoundedStack | tuple[int, tuple[str, ...]]:
	match bound:
		case BoundedStack():
			return bound
		case UnboundedStack(at_least_bytes=at_least, unmeasured=unmeasured):
			return at_least, unmeasured
		case _ as unreachable:
			assert_never(unreachable)


@pytest.mark.image
def test_the_main_stack_nests_on_the_reset_path_at_most_one_exception_per_priority_level(
	zephyr_fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	artifacts = zephyr_fixtures / "sensor-threads"
	stack(artifacts, artifacts / "zephyr" / "zephyr.elf", path="z_interrupt_stacks")
	assert capsys.readouterr().out.splitlines()[1:] == [
		"z_interrupt_stacks: unbounded, at least 10592 bytes (recursion: 50, measured: 13, nested exceptions: 10, priority levels: 8 (devicetree), exception frame: 36 bytes each, stack: 2048 bytes)",
		"(exception 1) __start +1016 = 1016 bytes",
		"(exception 4) z_arm_hard_fault +36 +1048 = 2100 bytes",
		"(exception 5) z_arm_hard_fault +36 +1048 = 3184 bytes",
		"(exception 6) z_arm_hard_fault +36 +1048 = 4268 bytes",
		"(exception 12) z_arm_hard_fault +36 +1048 = 5352 bytes",
		"(exception 15) sys_clock_isr +36 +1040 = 6428 bytes",
		"(exception 16) _isr_wrapper +36 +976 = 7440 bytes",
		"(exception 17) _isr_wrapper +36 +976 = 8452 bytes",
		"(exception 18) _isr_wrapper +36 +976 = 9464 bytes",
		"(exception 3) z_arm_hard_fault +36 +1048 = 10548 bytes",
		"(exception 2) z_arm_nmi +36 +8 = 10592 bytes",
	]


def _nesting_levels(out: str) -> tuple[int, LevelSource, int]:
	(nesting,) = (
		row.nesting
		for row in msgspec.json.decode(out, type=tuple[StackEntryReport, ...])
		if row.nesting is not None
	)
	return nesting.priority_levels, nesting.priority_levels_from, len(nesting.chain)


@pytest.mark.image
def test_without_its_devicetree_the_main_stack_takes_its_levels_from_the_vector_table(
	zephyr_fixtures: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	elf = tmp_path / "zephyr.elf"
	elf.write_bytes((zephyr_fixtures / "hello" / "zephyr" / "zephyr.elf").read_bytes())
	stack(tmp_path, elf, json=True)
	without = _nesting_levels(capsys.readouterr().out)
	(tmp_path / "zephyr.dts").write_bytes(
		(zephyr_fixtures / "hello" / "zephyr" / "zephyr.dts").read_bytes()
	)
	stack(tmp_path, elf, json=True)
	assert (without, _nesting_levels(capsys.readouterr().out)) == (
		(50, LevelSource.VECTOR_TABLE, 2 + 50),
		(8, LevelSource.DEVICETREE, 2 + 8),
	)
