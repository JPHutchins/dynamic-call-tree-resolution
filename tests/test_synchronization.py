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


@pytest.fixture(scope="module")
def used(zephyr_fixtures: Path) -> dict[str, int]:
	"""Both threads have run their loop several times by the sixth sleep."""
	return dict(
		zip(
			_STACK_BUFFERS,
			zephyr_used_stacks(
				ZephyrMachine.LM3S6965EVB,
				zephyr_fixtures / "synchronization" / "zephyr" / "zephyr.elf",
				GdbStop("z_impl_k_sleep_ticks", (), ignore=5),
				tuple(
					PaintedStack(f"&{buffer}", f"sizeof({buffer})")
					for buffer in _STACK_BUFFERS.values()
				),
			),
			strict=True,
		)
	)


@pytest.mark.parametrize(
	("assume_upward_frame_writes", "expected"),
	[
		pytest.param(False, ["thread_b", "z_idle_threads"], id="by default"),
		pytest.param(
			True,
			["thread_b", "z_idle_threads", "z_interrupt_stacks", "z_main_thread"],
			id="assuming upward frame writes",
		),
	],
)
def test_each_bounded_stack_row_holds_what_its_painted_stack_shows_used(
	zephyr_fixtures: Path,
	used: dict[str, int],
	capsys: pytest.CaptureFixture[str],
	assume_upward_frame_writes: bool,
	expected: list[str],
) -> None:
	artifacts = zephyr_fixtures / "synchronization"
	stack(
		artifacts,
		artifacts / "zephyr" / "zephyr.elf",
		json=True,
		assume_upward_frame_writes=assume_upward_frame_writes,
	)
	bounded = {
		row.entry: row.bound.bytes
		for row in msgspec.json.decode(capsys.readouterr().out, type=tuple[StackEntryReport, ...])
		if row.entry in _STACK_BUFFERS and isinstance(row.bound, BoundedStack)
	}
	assert (
		sorted(bounded),
		[entry for entry, depth in bounded.items() if used[entry] > depth],
	) == (expected, [])
