# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""RTOS models, and the thread entries they start with known arguments."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
from salix import replace

from dynamic_call_tree_resolution import (
	Address,
	LinkReference,
	Machine,
	ReferenceKind,
	build_report,
	load,
	resolve,
)
from dynamic_call_tree_resolution.cli import analyze, stack
from dynamic_call_tree_resolution.model import BARE_METAL, RtosModel, ThreadCreation
from dynamic_call_tree_resolution.points_to import pointer_at
from dynamic_call_tree_resolution.rtos import RtosChoice, rtos_model, zephyr
from dynamic_call_tree_resolution.vsa.fallback import referenced_only_at, referrers
from tests.programs import build_program
from tests.toolchains import zephyr_console_cortex_m3

if TYPE_CHECKING:
	from collections.abc import Callable

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


@pytest.mark.parametrize(
	("slots", "seeded"),
	[
		pytest.param((0x2000,), frozenset({Address(0x1001)}), id="only in its record"),
		pytest.param((0x2000, 0x2100), frozenset[Address](), id="also outside its record"),
	],
)
def test_an_entry_the_linker_also_references_outside_its_record_is_not_seeded(
	slots: tuple[int, ...], seeded: frozenset[Address]
) -> None:
	program = replace(
		build_program(
			Machine.EM_ARM,
			(("entry", 0x1001, 2),),
			objects=(("record", 0x2000, (0x1001).to_bytes(4, "little")),),
			sections={0x1000: bytes.fromhex("00bf")},
			pointer_size=4,
		),
		link_references=tuple(
			LinkReference(
				slot=Address(slot),
				symbol="entry",
				value=Address(0x1001),
				kind=ReferenceKind.ADDRESS,
			)
			for slot in slots
		),
	)
	assert (
		referenced_only_at(
			referrers(program, frozenset({Address(0x1001)})),
			{Address(0x1001): frozenset({Address(0x2000)})},
		)
		== seeded
	)


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
		name="zephyr",
		evidence=("z_thread_entry", "struct _static_thread_data"),
		threads=(),
		trampoline="z_thread_entry",
		creation=ThreadCreation(
			frame_builders=("arch_new_thread", "arch_switch_to_main_thread"),
			setup="z_setup_new_thread",
			static_start="z_init_static_threads",
			static_entries=frozenset(),
		),
	)


def _trampoline_targets(program: Program, model: RtosModel) -> list[str]:
	resolution = resolve(program, model)
	(site,) = (
		site
		for site in build_report(program, resolution.assignments, resolution.sites).call_sites
		if site.caller == "z_thread_entry"
	)
	return [candidate.name for candidate in site.candidates]


@pytest.mark.image
@pytest.mark.parametrize(
	("elf", "targets"),
	[
		pytest.param("hello/zephyr/zephyr.elf", ["bg_thread_main", "idle"], id="hello"),
		pytest.param(
			"sensor-two-impl/zephyr/zephyr.elf",
			["motion_thread", "thermal_thread", "bg_thread_main", "idle"],
			id="sensor-two-impl",
		),
		pytest.param(
			"sensor-threads/zephyr/zephyr.elf",
			["bg_thread_main", "sensor_thread", "idle"],
			id="sensor-threads",
		),
		pytest.param(
			"synchronization/zephyr/zephyr.elf",
			["thread_a_entry_point", "thread_b_entry_point", "bg_thread_main", "idle"],
			id="synchronization, with a k_thread_create thread",
		),
		pytest.param("counter-su/zephyr/zephyr.exe", [], id="counter-su calls it directly"),
	],
)
def test_the_call_that_starts_every_thread_goes_to_each_entry_its_creations_pass(
	zephyr_fixtures: Path, elf: str, targets: list[str]
) -> None:
	program = load(zephyr_fixtures / elf)
	assert _trampoline_targets(program, rtos_model(program, RtosChoice.AUTO)) == targets


@pytest.mark.image
def test_each_thread_qemu_runs_is_a_target_of_the_call_that_starts_every_thread(
	zephyr_fixtures: Path,
) -> None:
	elf = zephyr_fixtures / "synchronization" / "zephyr" / "zephyr.elf"
	program = load(elf)
	ran = frozenset(line.split(":")[0] for line in zephyr_console_cortex_m3(elf, 3)[1:])
	assert (
		ran,
		frozenset(f"{thread}_entry_point" for thread in ran)
		<= frozenset(_trampoline_targets(program, rtos_model(program, RtosChoice.AUTO))),
	) == (frozenset({"thread_a", "thread_b"}), True)


def _named(program: Program, name: str) -> Address:
	(address,) = (
		address for address, function in program.functions.items() if function.name == name
	)
	return address


def _builder_with_two_entries(program: Program, model: RtosModel) -> tuple[Program, RtosModel]:
	builder = program.functions[_named(program, "arch_new_thread")]
	assert builder.signature is not None
	return (
		replace(
			program,
			functions={
				**program.functions,
				builder.address: replace(
					builder,
					signature=replace(
						builder.signature,
						parameters=(*builder.signature.parameters, "function pointer"),
					),
				),
			},
		),
		model,
	)


def _trampoline_held_elsewhere(program: Program, model: RtosModel) -> tuple[Program, RtosModel]:
	(record, *_) = (
		data_object.address
		for data_object in program.objects.values()
		if data_object.name.startswith("_k_thread_data_")
	)
	return (
		replace(
			program,
			link_references=(
				*program.link_references,
				LinkReference(
					slot=record,
					symbol="z_thread_entry",
					value=_named(program, "z_thread_entry"),
					kind=ReferenceKind.ADDRESS,
				),
			),
		),
		model,
	)


def _records_incomplete(program: Program, model: RtosModel) -> tuple[Program, RtosModel]:
	assert model.creation is not None
	return program, replace(model, creation=replace(model.creation, static_entries=None))


@pytest.mark.image
@pytest.mark.parametrize(
	"perturbed",
	[
		pytest.param(_builder_with_two_entries, id="a frame builder with two entry parameters"),
		pytest.param(_trampoline_held_elsewhere, id="the trampoline held outside its builders"),
		pytest.param(_records_incomplete, id="the static thread records incomplete"),
	],
)
def test_the_call_that_starts_every_thread_stays_unresolved_when_a_creation_gate_fails(
	zephyr_fixtures: Path,
	perturbed: Callable[[Program, RtosModel], tuple[Program, RtosModel]],
) -> None:
	program = load(zephyr_fixtures / "sensor-threads" / "zephyr" / "zephyr.elf")
	assert _trampoline_targets(*perturbed(program, rtos_model(program, RtosChoice.AUTO))) == []


@pytest.mark.image
def test_a_static_thread_start_compiled_out_of_line_scopes_the_records_by_its_own_span(
	zephyr_fixtures: Path,
) -> None:
	program = load(zephyr_fixtures / "sensor-threads" / "zephyr" / "zephyr.elf")
	model = rtos_model(program, RtosChoice.AUTO)
	assert model.creation is not None
	assert _trampoline_targets(
		replace(program, inlined={}),
		replace(model, creation=replace(model.creation, static_start="bg_thread_main")),
	) == ["bg_thread_main", "sensor_thread", "idle"]


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


@pytest.mark.image
def test_cli_stack_reports_each_sensor_thread_in_place_of_its_entry(
	zephyr_fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	artifacts = zephyr_fixtures / "sensor-two-impl"
	stack(artifacts, artifacts / "zephyr" / "zephyr.elf")
	entries = {line.split(":")[0] for line in capsys.readouterr().out.splitlines()[1:]}
	stack(artifacts, artifacts / "zephyr" / "zephyr.elf", path="thermal_tid")
	steps = capsys.readouterr().out.splitlines()[2:4]
	stack(artifacts, artifacts / "zephyr" / "zephyr.elf", rtos=RtosChoice.NONE)
	bare_metal = {line.split(":")[0] for line in capsys.readouterr().out.splitlines()[1:]}
	threads_and_entries = {"motion_tid", "thermal_tid", "motion_thread", "thermal_thread"}
	assert (
		sorted(entries & threads_and_entries),
		steps,
		sorted(bare_metal & threads_and_entries),
	) == (
		["motion_tid", "thermal_tid"],
		["thermal_tid +8 = 8 bytes", "thermal_thread +8 = 16 bytes via thread record"],
		["motion_thread", "thermal_thread"],
	)
