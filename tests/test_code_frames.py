# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Frames measured from the code of functions without a ``.su`` record (#195, #214)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import pytest

from dynamic_call_tree_resolution import Address, StackEntryReport, UnboundedStack, load
from dynamic_call_tree_resolution.cli import stack
from dynamic_call_tree_resolution.vsa.frames import code_depths, code_measure
from tests.toolchains import FIXTURES, build_cortex_m3

if TYPE_CHECKING:
	from pathlib import Path


def _depths(elf: Path, names: tuple[str, ...]) -> dict[str, int | None]:
	program = load(elf)
	depths = code_depths(
		code_measure(program),
		frozenset(start for name in names for start in program.symbol_addresses[name]),
	)
	return {name: depths.get(start) for name in names for start in program.symbol_addresses[name]}


def test_a_functions_depth_is_what_its_code_moves_the_stack_by_and_its_callees_add(
	tmp_path: Path,
) -> None:
	assert _depths(
		build_cortex_m3((FIXTURES / "code_frames.c",), tmp_path / "image.elf", "-g", "-O2"),
		(
			"leaf_push",
			"calls_leaf",
			"tail_to_leaf",
			"unsized",
			"variable_sub",
			"indirect",
			"self_call",
		),
	) == {
		"leaf_push": 20,
		"calls_leaf": 8 + 20,
		"tail_to_leaf": 20,
		"unsized": 12,
		"variable_sub": None,
		"indirect": None,
		"self_call": None,
	}


def test_an_address_that_starts_no_function_has_no_depth(tmp_path: Path) -> None:
	program = load(
		build_cortex_m3((FIXTURES / "code_frames.c",), tmp_path / "image.elf", "-g", "-O2")
	)
	assert (
		code_depths(
			code_measure(program),
			frozenset(Address(start + 2) for start in program.symbol_addresses["leaf_push"]),
		)
		== {}
	)


@pytest.mark.image
def test_the_library_code_the_sensor_threads_reach_measures_as_its_disassembly_counts(
	zephyr_fixtures: Path,
) -> None:
	assert _depths(
		zephyr_fixtures / "sensor-threads" / "zephyr" / "zephyr.elf",
		(
			"memset",
			"strcmp",
			"__aeabi_memcpy8",
			"__udivmoddi4",
			"__aeabi_idiv0",
			"__aeabi_ldivmod",
			"z_arm_pendsv",
			"z_SysNmiOnReset",
			"__l_vfprintf",
			"__start",
			"z_arm_svc",
			"z_arm_hard_fault",
		),
	) == {
		"memset": 0,
		"strcmp": 0,
		"__aeabi_memcpy8": 8,
		"__udivmoddi4": 32,
		"__aeabi_idiv0": 0,
		"__aeabi_ldivmod": 16 + 32,
		"z_arm_pendsv": 0,
		"z_SysNmiOnReset": 0,
		"__l_vfprintf": None,
		"__start": None,
		"z_arm_svc": None,
		"z_arm_hard_fault": None,
	}


def test_a_su_record_wins_over_the_code_and_only_callees_without_one_are_measured(
	tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	elf = build_cortex_m3((FIXTURES / "code_frames.c",), tmp_path / "image.elf", "-g", "-O2")
	(tmp_path / "main.c.ci").write_text(
		'graph: { title: "main.c"\n'
		'node: { title: "main" label: "main\\nmain.c:1:5\\n16 bytes (static)" }\n'
		'edge: { sourcename: "main" targetname: "leaf_push" }\n'
		'edge: { sourcename: "main" targetname: "calls_leaf" }\n'
		"}\n"
	)
	(tmp_path / "main.c.su").write_text(
		"main.c:1:5:main\t16\tstatic\nmain.c:2:6:leaf_push\t4\tstatic\n"
	)
	stack(tmp_path, elf=elf, path="main")
	assert capsys.readouterr().out.splitlines()[1:] == [
		"main: 44 bytes (measured: 1)",
		"main +16 = 16 bytes",
		"calls_leaf +28 = 44 bytes via static (measured)",
	]


def test_a_register_call_its_ci_record_leaves_out_joins_the_graph_from_the_code(
	tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	elf = build_cortex_m3((FIXTURES / "code_frames.c",), tmp_path / "image.elf", "-g", "-O2")
	(tmp_path / "main.c.ci").write_text(
		'graph: { edge: { sourcename: "main" targetname: "indirect" } }\n'
	)
	(tmp_path / "main.c.su").write_text(
		"main.c:1:5:main\t16\tstatic\nmain.c:2:6:indirect\t8\tstatic\n"
	)
	stack(tmp_path, elf=elf, path="main")
	assert capsys.readouterr().out.splitlines()[1:] == [
		"main: 44 bytes (measured: 1)",
		"main +16 = 16 bytes",
		"indirect +8 = 24 bytes via static",
		"leaf_push +20 = 44 bytes via indirect: candidate (measured)",
	]


def test_a_function_that_calls_through_a_register_counts_its_own_frame_and_its_site(
	tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	elf = build_cortex_m3(
		(FIXTURES / "reproducers" / "own_frame.c",), tmp_path / "image.elf", "-O2", "-g"
	)
	build_directory = tmp_path / "build"
	build_directory.mkdir()
	stack(build_directory, elf, path="reset")
	assert capsys.readouterr().out.splitlines()[1:] == [
		"reset: unbounded, at least 64 bytes (unmeasured: 1, measured: 4)",
		"reset +24 = 24 bytes (measured)",
		"main +8 = 32 bytes via binary (measured)",
		"dispatch +32 = 64 bytes via binary (measured)",
		"handler +0 = 64 bytes via indirect: fallback (measured)",
	]


def test_a_function_that_branches_through_a_register_stays_unmeasured(
	tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	elf = build_cortex_m3(
		(FIXTURES / "reproducers" / "own_frame.c",), tmp_path / "image.elf", "-O2", "-g"
	)
	build_directory = tmp_path / "build"
	build_directory.mkdir()
	stack(build_directory, elf, json=True)
	assert [
		(row.bound.unmeasured, row.bound.measured)
		for row in msgspec.json.decode(capsys.readouterr().out, type=tuple[StackEntryReport, ...])
		if row.entry == "reset" and isinstance(row.bound, UnboundedStack)
	] == [(("forward",), ("dispatch", "handler", "main", "reset"))]


@pytest.mark.image
def test_library_code_without_su_adds_its_own_frame_and_its_register_calls_to_a_path(
	zephyr_fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	artifacts = zephyr_fixtures / "sensor-threads"
	stack(artifacts, artifacts / "zephyr" / "zephyr.elf", path="boot_banner")
	assert capsys.readouterr().out.splitlines()[1:8] == [
		"boot_banner: unbounded, at least 960 bytes (recursion: 50, measured: 8)",
		"boot_banner +8 = 8 bytes (recursion)",
		"printk +16 = 24 bytes via static (recursion)",
		"vprintk +0 = 24 bytes via static (recursion)",
		"vprintk_core +32 = 56 bytes via static (recursion)",
		"__l_vfprintf +80 = 136 bytes via static (measured, recursion)",
		"adt7420_attr_get +16 = 152 bytes via indirect: fallback (recursion)",
	]
