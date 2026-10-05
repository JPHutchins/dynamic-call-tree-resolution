# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The sensor-threads fixture: one entry for two static threads, bound to sensors on two buses."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import msgspec
import pytest
from salix import Struct

from dynamic_call_tree_resolution import build_report, load, resolve
from dynamic_call_tree_resolution.cli import stack
from dynamic_call_tree_resolution.report import BoundedStack, StackEntryReport
from dynamic_call_tree_resolution.rtos import RtosChoice, rtos_model
from tests.toolchains import zephyr_console_cortex_m3

if TYPE_CHECKING:
	from dynamic_call_tree_resolution.call_sites import ProgramResolution
	from dynamic_call_tree_resolution.model import Program

pytestmark = pytest.mark.image


QEMU_CONSOLE = (
	"*** Booting Zephyr OS build e560c91b1f74 ***",
	"thermal_tid unused 840 of 1024",
	"motion_tid unused 864 of 1024",
	"done",
)


def test_each_thread_reports_its_own_stack_high_water_on_qemu(zephyr_fixtures: Path) -> None:
	assert (
		zephyr_console_cortex_m3(zephyr_fixtures / "sensor-threads" / "zephyr" / "zephyr.elf", 4)
		== QEMU_CONSOLE
	)


def test_motion_tids_measured_high_water_is_within_its_bound_with_the_exception_frame(
	zephyr_fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	artifacts = zephyr_fixtures / "sensor-threads"
	stack(artifacts, artifacts / "zephyr" / "zephyr.elf", narrow_by_field=True, json=True)
	used = {
		thread: int(size) - int(unused)
		for line in QEMU_CONSOLE
		if " unused " in line
		for thread, _, unused, _, size in (line.split(),)
	}
	assert next(
		(
			used["motion_tid"],
			row.bound,
			row.bound.bytes - used["motion_tid"] if isinstance(row.bound, BoundedStack) else None,
		)
		for row in msgspec.json.decode(capsys.readouterr().out, type=tuple[StackEntryReport, ...])
		if row.entry == "motion_tid"
	) == (
		160,
		BoundedStack(
			bytes=184 + 36, measured=("__aeabi_ldivmod", "memset"), exception_frame_bytes=36
		),
		60,
	)


def test_the_readme_quotes_each_threads_high_water_as_qemu_prints_it() -> None:
	assert [
		line
		for line in QEMU_CONSOLE
		if " unused " in line
		and line
		not in {
			readme_line.strip()
			for readme_line in (Path(__file__).parent.parent / "README.md").read_text().splitlines()
		}
	] == []


def test_each_threads_stack_follows_its_own_driver_until_its_bus_emulator(
	zephyr_fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	artifacts = zephyr_fixtures / "sensor-threads"
	stack(artifacts, artifacts / "zephyr" / "zephyr.elf", path="thermal_tid")
	thermal = capsys.readouterr().out.splitlines()[1:8]
	stack(artifacts, artifacts / "zephyr" / "zephyr.elf", path="motion_tid")
	assert (thermal, capsys.readouterr().out.splitlines()[1:10]) == (
		[
			"thermal_tid: unbounded, at least 1092 bytes (recursion: 43, unmeasured: 4, measured: 6, exception frame: 36 bytes)",
			"thermal_tid +8 = 8 bytes",
			"sensor_thread +32 = 40 bytes via thread record",
			"adt7420_sample_fetch +32 = 72 bytes via indirect: candidate",
			"i2c_write_read +32 = 104 bytes via static",
			"i2c_emul_transfer +32 = 136 bytes via indirect: candidate",
			"adt7420_init +8 = 144 bytes via indirect: fallback (recursion)",
		],
		[
			"motion_tid: unbounded, at least 1104 bytes (recursion: 43, unmeasured: 4, measured: 6, exception frame: 36 bytes)",
			"motion_tid +8 = 8 bytes",
			"sensor_thread +32 = 40 bytes via thread record",
			"bmi160_sample_fetch +24 = 64 bytes via indirect: candidate",
			"bmi160_byte_read +0 = 64 bytes via static",
			"bmi160_read +4 = 68 bytes via static",
			"bmi160_read_spi +48 = 116 bytes via indirect: candidate",
			"spi_emul_io +32 = 148 bytes via indirect: candidate",
			"adt7420_init +8 = 156 bytes via indirect: fallback (recursion)",
		],
	)


class _Resolved(Struct):
	program: Program
	resolution: ProgramResolution


@pytest.fixture(scope="module")
def resolved(zephyr_fixtures: Path) -> _Resolved:
	program = load(zephyr_fixtures / "sensor-threads" / "zephyr" / "zephyr.elf")
	return _Resolved(
		program=program, resolution=resolve(program, rtos_model(program, RtosChoice.AUTO))
	)


def test_each_thread_runs_only_its_own_driver(resolved: _Resolved) -> None:
	assert {
		name: sorted(
			function
			for function in (resolved.program.functions[address].name for address in thread.reached)
			if function.startswith(("adt7420_", "bmi160_"))
		)
		for name, thread in resolved.resolution.threads.items()
	} == {
		"thermal_tid": ["adt7420_channel_get", "adt7420_sample_fetch"],
		"motion_tid": [
			"bmi160_byte_read",
			"bmi160_channel_convert",
			"bmi160_channel_get",
			"bmi160_read",
			"bmi160_read_spi",
			"bmi160_sample_fetch",
		],
	}


def test_each_threads_sites_resolve_to_its_own_driver_until_the_bus_emulator(
	resolved: _Resolved,
) -> None:
	assert {
		name: [
			f"{site.caller}: {', '.join(candidate.name for candidate in site.candidates) or '<unresolved>'}"
			for site in build_report(
				resolved.program, resolved.resolution.assignments, thread.sites
			).call_sites
		]
		for name, thread in resolved.resolution.threads.items()
	} == {
		"thermal_tid": [
			"z_thread_entry: sensor_thread",
			"sensor_thread: adt7420_sample_fetch",
			"sensor_thread: adt7420_channel_get",
			"i2c_emul_transfer: <unresolved>",
			"i2c_emul_transfer: <unresolved>",
			"i2c_emul_transfer: <unresolved>",
			"i2c_write_read.constprop.0: i2c_emul_transfer",
		],
		"motion_tid": [
			"z_thread_entry: sensor_thread",
			"sensor_thread: bmi160_sample_fetch",
			"sensor_thread: bmi160_channel_get",
			"bmi160_read_spi: spi_emul_io",
			"bmi160_read: bmi160_read_spi",
			"spi_emul_io: <unresolved>",
			"spi_emul_io: <unresolved>",
		],
	}
