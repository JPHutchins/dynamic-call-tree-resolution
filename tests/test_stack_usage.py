# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Tests for :mod:`dynamic_call_tree_resolution.stack_usage`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dynamic_call_tree_resolution import StackUsage
from dynamic_call_tree_resolution.stack_usage import load_stack_usages, parse_stack_usage

if TYPE_CHECKING:
	from pathlib import Path

SU_CONTENT = """\
/home/src/canbus.c:180:6:can_send\t24\tstatic
/home/src/canbus.c:12:5:rx_cb\t0\tstatic
/home/src/driver.c:99:1:z_impl_can_send\t16\tdynamic
"""


def test_parse_stack_usage(tmp_path: Path) -> None:
	stack_file = tmp_path / "canbus.c.su"
	stack_file.write_text(SU_CONTENT)
	assert parse_stack_usage(stack_file) == (
		StackUsage(function="can_send", bytes=24, bounded=True),
		StackUsage(function="rx_cb", bytes=0, bounded=True),
		StackUsage(function="z_impl_can_send", bytes=16, bounded=False),
	)


def test_parse_stack_usage_rejects_malformed_records(tmp_path: Path) -> None:
	stack_file = tmp_path / "bad.su"
	stack_file.write_text("can_send\t24\n")
	with pytest.raises(ValueError, match="not enough values to unpack"):
		parse_stack_usage(stack_file)


def test_parse_stack_usage_keeps_bounded_dynamic_frames_bounded(tmp_path: Path) -> None:
	stack_file = tmp_path / "bounded.su"
	stack_file.write_text("bounded.c:5:1:vla_frame\t48\tdynamic,bounded\n")
	assert parse_stack_usage(stack_file) == (
		StackUsage(function="vla_frame", bytes=48, bounded=True),
	)


def test_parse_stack_usage_rejects_unknown_qualifiers(tmp_path: Path) -> None:
	stack_file = tmp_path / "unknown.su"
	stack_file.write_text("unknown.c:5:1:frame\t48\tbounded\n")
	with pytest.raises(ValueError, match=r"unknown \.su qualifier 'bounded'"):
		parse_stack_usage(stack_file)


def test_parse_stack_usage_reads_clang_records_without_a_column(tmp_path: Path) -> None:
	stack_file = tmp_path / "vla.c.su"
	stack_file.write_text("vla.c:1:big\t64\tstatic\nvla.c:7:small\t8\tstatic\n")
	assert parse_stack_usage(stack_file) == (
		StackUsage(function="big", bytes=64, bounded=True),
		StackUsage(function="small", bytes=8, bounded=True),
	)


def test_parse_stack_usage_rejects_a_location_without_a_line(tmp_path: Path) -> None:
	stack_file = tmp_path / "bad.su"
	stack_file.write_text("big\t64\tstatic\n")
	with pytest.raises(ValueError, match=r"malformed \.su location 'big'"):
		parse_stack_usage(stack_file)


def test_load_stack_usages_collects_all_su_files(tmp_path: Path) -> None:
	build_directory = tmp_path / "build"
	nested = build_directory / "zephyr" / "CMakeFiles" / "zephyr.dir"
	nested.mkdir(parents=True)
	(nested / "canbus.c.obj.su").write_text("canbus.c:180:6:can_send\t24\tstatic\n")
	(nested / "counter.c.obj.su").write_text("main.c:1:1:main\t48\tstatic\n")
	assert load_stack_usages(build_directory) == (
		StackUsage(function="can_send", bytes=24, bounded=True),
		StackUsage(function="main", bytes=48, bounded=True),
	)
