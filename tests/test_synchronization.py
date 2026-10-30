# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The synchronization sample's stack rows against what its painted stacks show used on QEMU."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import msgspec
import pytest

from dynamic_call_tree_resolution.cli import stack
from dynamic_call_tree_resolution.report import BoundedStack, StackEntryReport
from tests.toolchains import GdbStop, PaintedStack, ZephyrMachine, zephyr_used_stacks

if TYPE_CHECKING:
	from pathlib import Path

pytestmark = pytest.mark.image

_STACK_BUFFERS: Final = {
	"thread_b": "_k_thread_stack_thread_b",
	"z_idle_threads": "z_idle_stacks",
	"z_main_thread": "z_main_stack",
	"z_interrupt_stacks": "z_interrupt_stacks",
}


def test_each_bounded_stack_row_holds_what_its_painted_stack_shows_used(
	zephyr_fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	"""Both threads have run their loop several times by the sixth sleep."""
	artifacts = zephyr_fixtures / "synchronization"
	elf = artifacts / "zephyr" / "zephyr.elf"
	stack(artifacts, elf, json=True)
	bounded = {
		row.entry: row.bound.bytes
		for row in msgspec.json.decode(capsys.readouterr().out, type=tuple[StackEntryReport, ...])
		if row.entry in _STACK_BUFFERS and isinstance(row.bound, BoundedStack)
	}
	used = dict(
		zip(
			_STACK_BUFFERS,
			zephyr_used_stacks(
				ZephyrMachine.LM3S6965EVB,
				elf,
				GdbStop("z_impl_k_sleep_ticks", (), ignore=5),
				tuple(
					PaintedStack(f"&{buffer}", f"sizeof({buffer})")
					for buffer in _STACK_BUFFERS.values()
				),
			),
			strict=True,
		)
	)
	assert (
		sorted(bounded),
		[entry for entry, depth in bounded.items() if used[entry] > depth],
	) == (["thread_b", "z_idle_threads"], [])
