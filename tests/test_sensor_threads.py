# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The sensor-threads fixture: one entry for two static threads, bound to sensors on two buses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

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
