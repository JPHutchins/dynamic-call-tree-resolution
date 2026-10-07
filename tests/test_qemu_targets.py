# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Every indirect call QEMU runs on an Arm Zephyr fixture goes where the analysis said (#75).

gdb reads each site's target from the register the site branches through, so a predicated
site whose condition fails would still record a call; no fixture site is predicated.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Final

import pytest
from salix import Struct

from dynamic_call_tree_resolution import Address, build_report, load, resolve
from dynamic_call_tree_resolution.descriptors import load_descriptors
from dynamic_call_tree_resolution.field_narrowing import field_narrowings
from dynamic_call_tree_resolution.loader import line_spans
from dynamic_call_tree_resolution.points_to import instruction_set_at, memory_at
from dynamic_call_tree_resolution.rtos import RtosChoice, rtos_model
from dynamic_call_tree_resolution.vsa.abi import disassemblers
from dynamic_call_tree_resolution.vsa.fallback import address_taken
from dynamic_call_tree_resolution.vsa.vectors import hardware_handlers
from tests.toolchains import IndirectCall, zephyr_indirect_calls_cortex_m3

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path

	from dynamic_call_tree_resolution.call_sites import ProgramResolution
	from dynamic_call_tree_resolution.field_narrowing import NarrowedSpan
	from dynamic_call_tree_resolution.model import CallSite, Program

pytestmark = pytest.mark.image

IMAGES: Final = ["hello", "sensor-threads", "sensor-two-impl", "synchronization"]
_GDB_REGISTERS: Final = {"ip": "r12", "sb": "r9", "sl": "r10", "fp": "r11"}
_GDB_REGISTER: Final = re.compile(r"r[0-9]|r1[0-2]")


class Run(Struct):
	"""One fixture, analyzed and run under gdb."""

	program: Program
	resolution: ProgramResolution
	narrowed_by_field: tuple[NarrowedSpan, ...]
	registers: Mapping[Address, str | None]
	"""The register each site branches through, or none when the site is not a register branch
	gdb can read."""
	calls: frozenset[IndirectCall]
	names: Mapping[int, str]


def _registers(program: Program, sites: tuple[CallSite, ...]) -> Mapping[Address, str | None]:
	return {
		site: next(
			(
				register
				for instruction in disassemblers()[instruction_set_at(program, site)].disasm(
					memory_at(program, site, 4), site, 1
				)
				if instruction.mnemonic in {"blx", "bx"}
				for register in (_GDB_REGISTERS.get(instruction.op_str, instruction.op_str),)
				if _GDB_REGISTER.fullmatch(register)
			),
			None,
		)
		for site in sorted({site.site_address for site in sites})
	}


def _run(fixtures: Path, scripts: Path, name: str) -> Run:
	elf = fixtures / name / "zephyr" / "zephyr.elf"
	program = load(elf)
	resolution = resolve(program, rtos_model(program, RtosChoice.AUTO))
	descriptors = load_descriptors(elf.parent / "descriptors.txt")
	registers = _registers(program, resolution.sites)
	return Run(
		program=program,
		resolution=resolution,
		narrowed_by_field=field_narrowings(
			program,
			descriptors,
			line_spans(elf, frozenset(site.location for site in descriptors.sites)),
		),
		registers=registers,
		calls=zephyr_indirect_calls_cortex_m3(
			elf,
			{site: register for site, register in registers.items() if register is not None},
			scripts / f"{name}.gdb",
			seconds=30,
			hits=3000,
		),
		names={start & ~1: function.name for start, function in program.functions.items()},
	)


@pytest.fixture(scope="module")
def runs(zephyr_fixtures: Path, tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, Run]:
	with ThreadPoolExecutor() as pool:
		return dict(
			zip(
				IMAGES,
				pool.map(partial(_run, zephyr_fixtures, tmp_path_factory.mktemp("gdb")), IMAGES),
				strict=True,
			)
		)


def _candidates(
	run: Run, sites: tuple[CallSite, ...], narrowed: tuple[NarrowedSpan, ...]
) -> Mapping[Address, frozenset[int]]:
	return {
		Address(site.site_address): frozenset(
			candidate.address & ~1 for candidate in site.candidates
		)
		for site in build_report(
			run.program, run.resolution.assignments, sites, narrowed_by_field=narrowed
		).call_sites
		if site.candidates and (site.field is not None or not narrowed)
	}


def _misses(
	run: Run, calls: frozenset[IndirectCall], allowed: Mapping[Address, frozenset[int]]
) -> list[tuple[str, str]]:
	return sorted(
		(f"{call.site:#x}", _name(run, call.target))
		for call in calls
		if call.site in allowed and call.target & ~1 not in allowed[call.site]
	)


def _name(run: Run, address: Address) -> str:
	return run.names.get(address & ~1, f"{address:#x}")


@pytest.mark.parametrize("name", IMAGES)
def test_every_site_branches_through_a_register_gdb_can_read(
	runs: Mapping[str, Run], name: str
) -> None:
	assert [
		f"{site:#x}" for site, register in runs[name].registers.items() if register is None
	] == []


@pytest.mark.parametrize("name", IMAGES)
def test_every_indirect_call_qemu_runs_goes_to_a_candidate_or_the_fallback(
	runs: Mapping[str, Run], name: str
) -> None:
	run = runs[name]
	candidates = _candidates(run, run.resolution.sites, ())
	fallback = frozenset(
		address & ~1 for address in address_taken(run.program) - hardware_handlers(run.program)
	)
	assert (
		bool(run.calls),
		_misses(
			run, run.calls, {call.site: candidates.get(call.site, fallback) for call in run.calls}
		),
	) == (True, [])


@pytest.mark.parametrize("name", IMAGES)
def test_every_indirect_call_qemu_runs_at_a_site_narrowed_by_field_goes_to_what_it_holds(
	runs: Mapping[str, Run], name: str
) -> None:
	run = runs[name]
	assert (
		_misses(run, run.calls, _candidates(run, run.resolution.sites, run.narrowed_by_field)) == []
	)


@pytest.mark.parametrize("name", ["sensor-threads", "sensor-two-impl", "synchronization"])
def test_every_indirect_call_a_static_thread_runs_goes_to_what_its_own_analysis_resolved(
	runs: Mapping[str, Run], name: str
) -> None:
	run = runs[name]
	objects = {
		data_object.name: data_object.address for data_object in run.program.objects.values()
	}
	assert [
		(thread, *miss)
		for thread, own in run.resolution.threads.items()
		for miss in _misses(
			run,
			frozenset(
				call for call in run.calls if call.thread == objects[f"_k_thread_obj_{thread}"]
			),
			_candidates(run, own.sites, ()),
		)
	] == []


def test_qemu_runs_each_sensor_threads_dispatch_on_its_own_thread(
	runs: Mapping[str, Run],
) -> None:
	run = runs["sensor-threads"]
	threads = {
		data_object.address: data_object.name.removeprefix("_k_thread_obj_")
		for data_object in run.program.objects.values()
	}
	assert {(_name(run, call.target), threads.get(call.thread)) for call in run.calls} >= {
		("sensor_thread", "motion_tid"),
		("sensor_thread", "thermal_tid"),
		("bmi160_sample_fetch", "motion_tid"),
		("adt7420_sample_fetch", "thermal_tid"),
		("bmi160_channel_get", "motion_tid"),
		("adt7420_channel_get", "thermal_tid"),
	}
