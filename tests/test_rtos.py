# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""RTOS models, and the thread entries they start with known arguments."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

from dynamic_call_tree_resolution import load, resolve
from dynamic_call_tree_resolution.cli import analyze
from dynamic_call_tree_resolution.model import BARE_METAL, RtosModel
from dynamic_call_tree_resolution.points_to import pointer_at
from dynamic_call_tree_resolution.rtos import RtosChoice, rtos_model, zephyr

if TYPE_CHECKING:
	from dynamic_call_tree_resolution.model import Program

PACKAGE: Final = Path(__file__).parent.parent / "src" / "dynamic_call_tree_resolution"

ADAPTERS: Final = "dynamic_call_tree_resolution.rtos"


def _imports(node: ast.AST) -> tuple[str, ...]:
	match node:
		case ast.Import(names=names):
			return tuple(alias.name for alias in names)
		case ast.ImportFrom(module=str() as module):
			return (module,)
		case _:
			return ()


def _imported(path: Path) -> frozenset[str]:
	return frozenset(
		name for node in ast.walk(ast.parse(path.read_text())) for name in _imports(node)
	)


def _imports_an_adapter(path: Path) -> bool:
	return any(name == ADAPTERS or name.startswith(f"{ADAPTERS}.") for name in _imported(path))


def test_only_the_cli_imports_an_rtos_adapter() -> None:
	assert [
		path.relative_to(PACKAGE).as_posix()
		for path in sorted(PACKAGE.rglob("*.py"))
		if "rtos" not in path.relative_to(PACKAGE).parts and _imports_an_adapter(path)
	] == ["cli.py"]


def test_an_image_without_zephyr_is_bare_metal(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["nopie"])
	assert (
		zephyr.detect(program),
		rtos_model(program, RtosChoice.AUTO),
		rtos_model(program, RtosChoice.NONE),
	) == (None, BARE_METAL, BARE_METAL)


def test_zephyr_skips_a_writable_record_and_one_without_an_entry(
	fixture_elfs: dict[str, Path],
) -> None:
	program = load(fixture_elfs["threads"])
	model = rtos_model(program, RtosChoice.ZEPHYR)
	assert (
		sorted(
			(thread.name, program.functions[thread.entry].name, thread.arguments[1:])
			for thread in model.threads
		),
		resolve(program, model).seeded,
	) == (
		[("escaped_tid", "escaped_thread", (0, 0)), ("rom_tid", "rom_thread", (0, 0))],
		frozenset({"rom_tid"}),
	)


def test_cli_analyze_says_which_thread_entries_are_seeded(
	fixture_elfs: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
	analyze(fixture_elfs["threads"])
	assert {
		line for line in capsys.readouterr().out.splitlines() if line.startswith("thread ")
	} == {
		"thread escaped_tid: escaped_thread (not seeded: its address is taken elsewhere)",
		"thread rom_tid: rom_thread (seeded from its record)",
	}


def test_choosing_zephyr_for_an_image_without_it_is_an_error(
	fixture_elfs: dict[str, Path],
) -> None:
	with pytest.raises(ValueError, match="no Zephyr"):
		rtos_model(load(fixture_elfs["nopie"]), RtosChoice.ZEPHYR)


@pytest.mark.image
def test_zephyr_is_detected_in_hello_world_which_defines_no_static_thread(
	zephyr_fixtures: Path,
) -> None:
	assert rtos_model(
		load(zephyr_fixtures / "hello" / "zephyr" / "zephyr.elf"), RtosChoice.AUTO
	) == RtosModel(
		name="zephyr", evidence=("z_thread_entry", "struct _static_thread_data"), threads=()
	)


@pytest.fixture(scope="module")
def sensor_program(zephyr_fixtures: Path) -> Program:
	return load(zephyr_fixtures / "sensor-two-impl" / "zephyr" / "zephyr.elf")


@pytest.mark.image
def test_zephyr_reads_each_static_thread_from_its_record(sensor_program: Program) -> None:
	model = rtos_model(sensor_program, RtosChoice.ZEPHYR)
	assert [
		(
			thread.name,
			sensor_program.functions[thread.entry].name,
			pointer_at(sensor_program, thread.entry_slot) == thread.entry,
			thread.arguments,
		)
		for thread in model.threads
	] == [
		("motion_tid", "motion_thread", True, (0, 0, 0)),
		("thermal_tid", "thermal_thread", True, (0, 0, 0)),
	]


@pytest.mark.image
def test_each_sensor_thread_entry_starts_from_its_record(sensor_program: Program) -> None:
	assert resolve(sensor_program, rtos_model(sensor_program, RtosChoice.AUTO)).seeded == frozenset(
		{
			"motion_tid",
			"thermal_tid",
		}
	)


@pytest.mark.image
def test_cli_analyze_names_the_rtos_and_its_threads_unless_told_none(
	zephyr_fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	elf = zephyr_fixtures / "sensor-two-impl" / "zephyr" / "zephyr.elf"
	analyze(elf)
	detected = capsys.readouterr().out.splitlines()
	analyze(elf, rtos=RtosChoice.NONE)
	assert (
		detected[:3],
		[
			line
			for line in capsys.readouterr().out.splitlines()
			if line.startswith(("rtos:", "thread "))
		],
	) == (
		[
			"rtos: zephyr (detected: z_thread_entry, struct _static_thread_data)",
			"thread motion_tid: motion_thread (seeded from its record)",
			"thread thermal_tid: thermal_thread (seeded from its record)",
		],
		[],
	)
