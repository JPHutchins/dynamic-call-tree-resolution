# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The benchmark measurements CI never runs still measure the analysis they name."""

import asyncio
import sys
from pathlib import Path

import msgspec
import pytest

from benchmarks.measurements import CpuTime, Work

pytestmark = pytest.mark.image


def _measured(kind: str, elf: Path) -> bytes:
	return asyncio.run(_measured_in_process(kind, elf))


async def _measured_in_process(kind: str, elf: Path) -> bytes:
	process = await asyncio.create_subprocess_exec(
		sys.executable,
		"-m",
		"benchmarks.measure",
		kind,
		str(elf),
		cwd=Path(__file__).parent.parent,
		stdout=asyncio.subprocess.PIPE,
	)
	stdout, _ = await process.communicate()
	assert await process.wait() == 0
	return stdout


def test_work_is_the_same_every_run_and_counts_each_function_analysis_and_transfer(
	zephyr_fixtures: Path,
) -> None:
	elf = zephyr_fixtures / "hello" / "zephyr" / "zephyr.elf"
	first = msgspec.json.decode(_measured("work", elf), type=Work)
	assert msgspec.json.decode(_measured("work", elf), type=Work) == first
	assert first.transfers > first.function_analyses > 0


def test_cpu_time_is_the_analysis_alone(zephyr_fixtures: Path) -> None:
	elf = zephyr_fixtures / "hello" / "zephyr" / "zephyr.elf"
	assert msgspec.json.decode(_measured("cpu", elf), type=CpuTime).cpu_seconds > 0


def test_a_measurement_with_a_field_its_type_lacks_is_refused() -> None:
	with pytest.raises(msgspec.ValidationError, match="unknown field"):
		msgspec.json.decode(
			b'{"transfers": 2, "function_analyses": 1, "cpu_seconds": 1}', type=Work
		)
