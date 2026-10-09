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
from tests.toolchains import zephyr_console_cortex_m3, zephyr_stack_pointers_cortex_m3

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


@pytest.mark.parametrize(
	("assume_no_recursion", "assume_unwritten", "provenance"),
	[
		pytest.param(
			("i2c_emul_transfer",),
			(),
			{"assumed_no_recursion": ("i2c_emul_transfer",)},
			id="assume no recursion",
		),
		pytest.param(
			(),
			("i2c_emul_cfg_0",),
			{"assumed_unwritten": ("i2c_emul_transfer",)},
			id="assume the I2C emulator's config unwritten",
		),
	],
)
def test_each_threads_measured_high_water_is_within_its_bound(
	zephyr_fixtures: Path,
	capsys: pytest.CaptureFixture[str],
	assume_no_recursion: tuple[str, ...],
	assume_unwritten: tuple[str, ...],
	provenance: dict[str, tuple[str, ...]],
) -> None:
	artifacts = zephyr_fixtures / "sensor-threads"
	stack(
		artifacts,
		artifacts / "zephyr" / "zephyr.elf",
		narrow_by_field=True,
		assume_no_recursion=assume_no_recursion,
		assume_unwritten=assume_unwritten,
		json=True,
	)
	used = {
		thread: int(size) - int(unused)
		for line in QEMU_CONSOLE
		if " unused " in line
		for thread, _, unused, _, size in (line.split(),)
	}
	assert {
		row.entry: (
			used[row.entry],
			row.bound,
			row.bound.bytes - used[row.entry] if isinstance(row.bound, BoundedStack) else None,
		)
		for row in msgspec.json.decode(capsys.readouterr().out, type=tuple[StackEntryReport, ...])
		if row.entry in used
	} == {
		"motion_tid": (
			160,
			BoundedStack(
				bytes=16 + 184 + 36,
				measured=("__aeabi_ldivmod", "__aeabi_read_tp", "memset"),
				stack_reservation_bytes=16,
				exception_frame_bytes=36,
				narrowed_by_field=("spi_emul_io",),
			),
			76,
		),
		"thermal_tid": (
			184,
			BoundedStack(
				bytes=16 + 184 + 36,
				measured=("__aeabi_ldivmod", "__aeabi_read_tp", "memset"),
				stack_reservation_bytes=16,
				exception_frame_bytes=36,
				narrowed_by_field=("i2c_emul_transfer",),
				**provenance,
			),
			52,
		),
	}


def test_the_header_names_an_object_assumed_unwritten_that_a_tracked_store_writes(
	zephyr_fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	artifacts = zephyr_fixtures / "sensor-threads"
	stack(
		artifacts,
		artifacts / "zephyr" / "zephyr.elf",
		assume_unwritten=("_stdout_hook", "i2c_emul_cfg_0"),
	)
	assert (
		" | assumed unwritten: _stdout_hook, i2c_emul_cfg_0"
		" | yet tracked stores write: _stdout_hook | "
	) in capsys.readouterr().out.splitlines()[0]


@pytest.mark.parametrize(
	("fixture", "name", "message"),
	[
		pytest.param(
			"sensor-threads",
			"no_such_object",
			"no data object is named no_such_object",
			id="an unknown name",
		),
		pytest.param(
			"synchronization",
			"__func__",
			"2 data objects are named __func__",
			id="a name two objects share",
		),
	],
)
def test_assume_unwritten_names_one_data_object(
	zephyr_fixtures: Path, fixture: str, name: str, message: str
) -> None:
	artifacts = zephyr_fixtures / fixture
	with pytest.raises(ValueError, match=message):
		stack(artifacts, artifacts / "zephyr" / "zephyr.elf", assume_unwritten=(name,))


def test_each_threads_stack_is_the_size_qemu_reports_and_its_margin_is_within_qemus_unused(
	zephyr_fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	artifacts = zephyr_fixtures / "sensor-threads"
	stack(
		artifacts,
		artifacts / "zephyr" / "zephyr.elf",
		narrow_by_field=True,
		assume_no_recursion=("i2c_emul_transfer",),
		json=True,
	)
	qemu = {
		thread: (int(size), int(unused))
		for line in QEMU_CONSOLE
		if " unused " in line
		for thread, _, unused, _, size in (line.split(),)
	}
	assert {
		row.entry: (
			row.stack_bytes,
			row.margin_bytes,
			row.margin_bytes is not None and row.margin_bytes <= qemu[row.entry][1],
		)
		for row in msgspec.json.decode(capsys.readouterr().out, type=tuple[StackEntryReport, ...])
		if row.entry in qemu
	} == {thread: (size, size - (16 + 184 + 36), True) for thread, (size, _) in qemu.items()}


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
			"thermal_tid: unbounded, at least 1180 bytes (recursion: 50, measured: 8, stack reservation: 16 bytes, exception frame: 36 bytes, stack: 1024 bytes)",
			"thermal_tid +8 = 8 bytes",
			"sensor_thread +32 = 40 bytes via thread record",
			"adt7420_sample_fetch +32 = 72 bytes via indirect: candidate",
			"i2c_write_read +32 = 104 bytes via static",
			"i2c_emul_transfer +32 = 136 bytes via indirect: candidate",
			"adt7420_init +8 = 144 bytes via indirect: fallback (recursion)",
		],
		[
			"motion_tid: unbounded, at least 1192 bytes (recursion: 50, measured: 8, stack reservation: 16 bytes, exception frame: 36 bytes, stack: 1024 bytes)",
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


def test_a_threads_path_ends_with_what_the_rtos_model_adds_to_its_code(
	zephyr_fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	artifacts = zephyr_fixtures / "sensor-threads"
	stack(artifacts, artifacts / "zephyr" / "zephyr.elf", narrow_by_field=True, path="motion_tid")
	assert capsys.readouterr().out.splitlines()[-3:] == [
		"elapsed@cortex_m_systick.c +8 = 184 bytes via static",
		"(stack reservation) +16 = 200 bytes",
		"(exception frame) +36 = 236 bytes",
	]


def test_the_main_and_idle_threads_are_rows_in_place_of_their_entries(
	zephyr_fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	artifacts = zephyr_fixtures / "sensor-threads"
	stack(artifacts, artifacts / "zephyr" / "zephyr.elf", json=True)
	rows = {
		row.entry: row.bound
		for row in msgspec.json.decode(capsys.readouterr().out, type=tuple[StackEntryReport, ...])
	}
	stack(artifacts, artifacts / "zephyr" / "zephyr.elf", rtos=RtosChoice.NONE, json=True)
	bare_metal = {
		row.entry
		for row in msgspec.json.decode(capsys.readouterr().out, type=tuple[StackEntryReport, ...])
	}
	threads_and_entries = {"z_main_thread", "z_idle_threads", "bg_thread_main", "idle"}
	assert (
		sorted(rows.keys() & threads_and_entries),
		sorted(bare_metal & threads_and_entries),
		rows.get("z_idle_threads"),
	) == (
		["z_idle_threads", "z_main_thread"],
		["bg_thread_main", "idle"],
		BoundedStack(
			bytes=16 + 184 + 36,
			measured=("__aeabi_read_tp",),
			stack_reservation_bytes=16,
			exception_frame_bytes=36,
		),
	)


def test_the_main_threads_path_starts_in_the_trampoline_and_ends_with_what_the_rtos_model_adds(
	zephyr_fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	artifacts = zephyr_fixtures / "sensor-threads"
	stack(artifacts, artifacts / "zephyr" / "zephyr.elf", path="z_main_thread")
	lines = capsys.readouterr().out.splitlines()
	assert (lines[1:4], lines[-2:]) == (
		[
			"z_main_thread: unbounded, at least 1044 bytes (recursion: 50, measured: 8, stack reservation: 16 bytes, exception frame: 36 bytes, stack: 1024 bytes)",
			"z_main_thread +8 = 8 bytes",
			"bg_thread_main +40 = 48 bytes via system thread (recursion)",
		],
		["(stack reservation) +16 = 1008 bytes", "(exception frame) +36 = 1044 bytes"],
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


def test_each_thread_starts_its_stack_reservation_below_its_stack_top_on_qemu(
	zephyr_fixtures: Path, resolved: _Resolved
) -> None:
	assert set(
		zephyr_stack_pointers_cortex_m3(
			zephyr_fixtures / "sensor-threads" / "zephyr" / "zephyr.elf", "*z_thread_entry", 3
		)
	) == {
		stack.address + stack.size - model.stack_reservation
		for model in (rtos_model(resolved.program, RtosChoice.AUTO),)
		for stack in resolved.program.objects.values()
		if stack.name
		in {"z_main_stack", "_k_thread_stack_motion_tid", "_k_thread_stack_thermal_tid"}
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
