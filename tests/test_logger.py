# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The logger fixture: Zephyr's deferred logging, whose callbacks unknown stores hide (#341)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from dynamic_call_tree_resolution import build_report, load, resolve
from dynamic_call_tree_resolution.cli.expansion import named_span
from dynamic_call_tree_resolution.rtos import RtosChoice, rtos_model

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path

pytestmark = pytest.mark.image

_EVERY_STORE_PLACED: Final = (
	"log_buffer.get_wlen",
	"log_buffer.notify_drop",
	"log_process_thread_timer.stop_fn",
	"timestamp_func",
)


def _sites(fixtures: Path, only_tracked_writes: tuple[str, ...]) -> Mapping[str, tuple[str, ...]]:
	program = load(fixtures / "logger" / "zephyr" / "zephyr.elf")
	resolution = resolve(
		program,
		rtos_model(program, RtosChoice.AUTO),
		only_tracked_writes=tuple(
			named_span(program, "--assume-only-tracked-writes", name)
			for name in only_tracked_writes
		),
	)
	return {
		f"{site.caller}@{site.site_address:#x}": (
			("<dead>",)
			if site.dead
			else tuple(sorted(candidate.name for candidate in site.candidates))
		)
		for site in build_report(program, resolution.assignments, resolution.sites).call_sites
	}


def test_unknown_stores_leave_24_of_the_loggers_59_sites_unresolved(zephyr_fixtures: Path) -> None:
	sites = _sites(zephyr_fixtures, ())
	assert (len(sites), sum(candidates == () for candidates in sites.values())) == (59, 24)


def test_keeping_the_logging_callbacks_through_unknown_stores_resolves_7_of_them(
	zephyr_fixtures: Path,
) -> None:
	before = _sites(zephyr_fixtures, ())
	after = _sites(zephyr_fixtures, _EVERY_STORE_PLACED)
	assert {site: after[site] for site in after if after[site] != before[site]} == {
		"z_log_msg_commit@0xe08": ("default_lf_get_timestamp", "dummy_timestamp", "timestamp_get"),
		"z_impl_k_timer_stop@0x3252": ("<dead>",),
		"drop_item_locked@0x3d94": ("log_msg_generic_get_wlen",),
		"mpsc_pbuf_alloc@0x3f32": ("z_log_notify_drop",),
		"mpsc_pbuf_commit@0x3faa": ("log_msg_generic_get_wlen",),
		"mpsc_pbuf_claim@0x402e": ("log_msg_generic_get_wlen",),
		"mpsc_pbuf_claim@0x405c": ("log_msg_generic_get_wlen",),
	}
