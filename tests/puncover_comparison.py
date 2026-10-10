# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The README's puncover comparison: puncover's static stack size next to dctr's stack bound,
for entries of the sensor-two-impl build, generated so the published table is tool output.

Run it inside ``nix develop``, which puts ``arm-none-eabi-*`` on PATH and exports
``DCTR_FIXTURES``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final, assert_never

import msgspec

from dynamic_call_tree_resolution.report import BoundedStack, StackPathReport, UnboundedStack
from tests.dctr import dctr

BUILD: Final = "sensor-two-impl"
ROWS: Final = (
	("sys_clock_isr", "sys_clock_isr"),
	("_isr_wrapper", "_isr_wrapper"),
	("motion_thread", "motion_tid"),
	("thermal_thread", "thermal_tid"),
	("main", "z_main_thread"),
	("sys_clock_driver_init", "sys_clock_driver_init"),
)
"""Each row's puncover function and the dctr entry that starts there; a thread's dctr entry is
its thread object, whose bound adds the RTOS model's stack reservation and exception frame."""
HEADER: Final = ("puncover entry", "puncover 0.8.0", "dctr entry", "dctr stack")


class _StackUsage(msgspec.Struct):
	max_static_stack_size: int


class _Tagged(msgspec.Struct):
	stack_report: dict[str, _StackUsage]


def comparison(fixtures: Path) -> str:
	build = fixtures / BUILD
	elf = build / "zephyr" / "zephyr.elf"
	puncover = _puncover(build, elf, tuple(function for function, _ in ROWS))
	rows = (
		HEADER,
		*(
			(function, _bytes(puncover.get(function)), entry, _bound(build, elf, entry))
			for function, entry in ROWS
		),
	)
	widths = tuple(max(len(row[column]) for row in rows) for column in range(len(HEADER)))
	return "".join(
		"  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
		+ "\n"
		for row in rows
	)


def _puncover(build: Path, elf: Path, functions: tuple[str, ...]) -> dict[str, int]:
	with tempfile.TemporaryDirectory() as directory:
		subprocess.run(
			[
				str(Path(sys.executable).with_name("puncover")),
				"--gcc-tools-base",
				_tools_prefix(),
				"--build-dir",
				str(build),
				"--non-interactive",
				"--generate-report",
				*(
					argument
					for function in functions
					for argument in ("--report-max-static-stack-usage", function)
				),
				str(elf),
			],
			cwd=directory,
			check=True,
			capture_output=True,
		)
		report = msgspec.json.decode(
			(Path(directory) / "report.json").read_bytes(), type=dict[str, _Tagged]
		)
	return {
		function: usage.max_static_stack_size
		for function, usage in report["no_tag"].stack_report.items()
	}


def _tools_prefix() -> str:
	gcc = shutil.which("arm-none-eabi-gcc")
	if gcc is None:
		raise RuntimeError("arm-none-eabi-gcc is not on PATH: run inside `nix develop`")
	return str(Path(gcc).with_name("arm-none-eabi-"))


def _bytes(size: int | None) -> str:
	return "-" if size is None else f"{size} bytes"


def _bound(build: Path, elf: Path, entry: str) -> str:
	match msgspec.json.decode(
		dctr("stack", str(build), "--elf", str(elf), "--path", entry, "--json"),
		type=StackPathReport,
	).bound:
		case BoundedStack(bytes=size):
			return f"{size} bytes"
		case UnboundedStack(at_least_bytes=size):
			return f"unbounded, at least {size} bytes"
		case _ as unreachable:
			assert_never(unreachable)


if __name__ == "__main__":
	print(comparison(Path(os.environ["DCTR_FIXTURES"])), end="")
