# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Invariants every fixture's analysis holds, whatever its numbers (#75, item 2)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from dynamic_call_tree_resolution import Address, build_report
from dynamic_call_tree_resolution.cli import elf_expansion
from dynamic_call_tree_resolution.identity import stack_names
from dynamic_call_tree_resolution.rtos import RtosChoice
from dynamic_call_tree_resolution.stack_analysis import frame_key
from dynamic_call_tree_resolution.vsa.abi import normalized

if TYPE_CHECKING:
	from pathlib import Path

pytestmark = pytest.mark.image

IMAGES: Final = [
	("hello", "zephyr.elf"),
	("sensor-threads", "zephyr.elf"),
	("sensor-two-impl", "zephyr.elf"),
	("synchronization", "zephyr.elf"),
	pytest.param(
		"counter-su",
		"zephyr.exe",
		marks=pytest.mark.xfail(
			strict=True,
			raises=AssertionError,
			reason="https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/228",
		),
	),
]


@pytest.mark.parametrize(("name", "elf"), IMAGES)
def test_every_candidate_analyze_reports_is_a_callee_of_its_site_in_the_stack_graph(
	zephyr_fixtures: Path, name: str, elf: str
) -> None:
	expansion = elf_expansion(
		zephyr_fixtures / name,
		zephyr_fixtures / name / "zephyr" / elf,
		narrow_by_signature=False,
		narrow_by_field=False,
		rtos=RtosChoice.AUTO,
	)
	edges = frozenset(
		(frame_key(edge.caller), frame_key(edge.callee)) for edge in expansion.expanded
	)
	names = stack_names(expansion.program)
	callers = {site.site_address: site.caller_address for site in expansion.resolution.sites}
	assert [
		(f"{site.caller}@{site.site_address:#x}", candidate.name)
		for site in build_report(
			expansion.program, expansion.resolution.assignments, expansion.resolution.sites
		).call_sites
		for candidate in site.candidates
		if (
			frame_key(
				names[normalized(callers[Address(site.site_address)], expansion.program.machine)]
			),
			frame_key(names[normalized(Address(candidate.address), expansion.program.machine)]),
		)
		not in edges
	] == []
