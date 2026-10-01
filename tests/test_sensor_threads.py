# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The sensor-threads fixture: one entry for two static threads, bound to sensors on two buses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dynamic_call_tree_resolution import build_report, load, resolve
from dynamic_call_tree_resolution.rtos import RtosChoice, rtos_model
from tests.toolchains import zephyr_console_cortex_m3

if TYPE_CHECKING:
	from pathlib import Path

pytestmark = pytest.mark.image


def test_each_thread_reports_its_own_stack_high_water_on_qemu(zephyr_fixtures: Path) -> None:
	assert zephyr_console_cortex_m3(
		zephyr_fixtures / "sensor-threads" / "zephyr" / "zephyr.elf", 4
	) == (
		"*** Booting Zephyr OS build e560c91b1f74 ***",
		"thermal_tid unused 840 of 1024",
		"motion_tid unused 864 of 1024",
		"done",
	)


def test_each_threads_own_analysis_reaches_only_its_own_driver_until_the_bus_emulator(
	zephyr_fixtures: Path,
) -> None:
	program = load(zephyr_fixtures / "sensor-threads" / "zephyr" / "zephyr.elf")
	resolution = resolve(program, rtos_model(program, RtosChoice.AUTO))
	assert {
		name: [
			f"{site.caller}: {', '.join(candidate.name for candidate in site.candidates) or '<unresolved>'}"
			for site in build_report(program, resolution.assignments, thread.sites).call_sites
		]
		for name, thread in resolution.threads.items()
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
