# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Invariants every fixture's analysis holds, whatever its numbers (#75, item 2)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from functools import partial
from typing import TYPE_CHECKING, Final, assert_never

import pytest

from dynamic_call_tree_resolution import Address, Bounded, StackReport, Unbounded, build_report
from dynamic_call_tree_resolution.call_sites import own_targets, per_caller_candidates
from dynamic_call_tree_resolution.cli import Expansion, elf_expansion
from dynamic_call_tree_resolution.descriptors import load_descriptors
from dynamic_call_tree_resolution.field_narrowing import field_narrowings
from dynamic_call_tree_resolution.identity import stack_names
from dynamic_call_tree_resolution.loader import line_spans
from dynamic_call_tree_resolution.rtos import RtosChoice
from dynamic_call_tree_resolution.stack_analysis import (
	INDIRECT_CALLEE,
	NULL_CALLEE,
	expand_indirect_calls,
	frame_key,
	worst_case_depths,
)
from dynamic_call_tree_resolution.vsa.abi import normalized

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path

	from dynamic_call_tree_resolution.field_narrowing import NarrowedSpan

pytestmark = pytest.mark.image

FIXTURES: Final = {
	"hello": "zephyr.elf",
	"sensor-threads": "zephyr.elf",
	"sensor-two-impl": "zephyr.elf",
	"synchronization": "zephyr.elf",
	"counter-su": "zephyr.exe",
}


class Narrowing(StrEnum):
	NONE = "none"
	FIELD = "field"
	SIGNATURE = "signature"


def _tracked(issue: int) -> pytest.MarkDecorator:
	return pytest.mark.xfail(
		strict=True,
		raises=AssertionError,
		reason=f"https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/{issue}",
	)


def _elf(fixtures: Path, name: str) -> Path:
	return fixtures / name / "zephyr" / FIXTURES[name]


def _expansion(fixtures: Path, name: str) -> Expansion:
	return elf_expansion(
		fixtures / name,
		_elf(fixtures, name),
		narrow_by_signature=False,
		narrow_by_field=False,
		rtos=RtosChoice.AUTO,
	)


@pytest.fixture(scope="module")
def expansions(zephyr_fixtures: Path) -> Mapping[str, Expansion]:
	with ThreadPoolExecutor() as pool:
		return dict(
			zip(FIXTURES, pool.map(partial(_expansion, zephyr_fixtures), FIXTURES), strict=True)
		)


@pytest.mark.parametrize(
	"name",
	[pytest.param(name, marks=_tracked(228) if name == "counter-su" else ()) for name in FIXTURES],
)
def test_every_candidate_analyze_reports_is_a_callee_of_its_site_in_the_stack_graph(
	expansions: Mapping[str, Expansion], name: str
) -> None:
	expansion = expansions[name]
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


def _narrowed_by_field(fixtures: Path, name: str, expansion: Expansion) -> tuple[NarrowedSpan, ...]:
	descriptors = load_descriptors(_elf(fixtures, name).parent / "descriptors.txt")
	return field_narrowings(
		expansion.program,
		descriptors,
		line_spans(_elf(fixtures, name), frozenset(site.location for site in descriptors.sites)),
	)


def _narrowed(
	fixtures: Path, name: str, expansion: Expansion, narrowing: Narrowing
) -> tuple[bool, tuple[NarrowedSpan, ...]]:
	match narrowing:
		case Narrowing.NONE:
			return False, ()
		case Narrowing.FIELD:
			return False, _narrowed_by_field(fixtures, name, expansion)
		case Narrowing.SIGNATURE:
			return True, ()
		case _ as unreachable:
			assert_never(unreachable)


@pytest.mark.parametrize(
	("name", "narrowing"),
	[
		(name, narrowing)
		for name in FIXTURES
		for narrowing in Narrowing
		if not (name == "counter-su" and narrowing is Narrowing.FIELD)
	],
)
def test_every_target_a_resolution_names_is_in_the_fallback_so_removing_it_never_lowers_a_bound(
	zephyr_fixtures: Path, expansions: Mapping[str, Expansion], name: str, narrowing: Narrowing
) -> None:
	expansion = expansions[name]
	names = stack_names(expansion.program)
	narrow_by_signature, narrowed_by_field = _narrowed(zephyr_fixtures, name, expansion, narrowing)
	targets_by_caller, fallback = per_caller_candidates(
		expansion.program,
		expansion.resolution.sites,
		expansion.resolution.assignments,
		narrow_by_signature=narrow_by_signature,
		narrowed_by_field=narrowed_by_field,
		names=names,
	)
	named = [
		(resolution, caller, target)
		for resolution, by_caller in (
			("global", targets_by_caller),
			*(
				(
					thread,
					own_targets(
						expansion.program,
						sites,
						expansion.resolution.assignments,
						names,
						narrowed_by_field,
						narrow_by_signature=narrow_by_signature,
					).targets_by_caller,
				)
				for thread, sites in expansion.resolution.threads.items()
			),
		)
		for caller, targets in by_caller.items()
		for target in targets - {INDIRECT_CALLEE, NULL_CALLEE}
	]
	assert (
		bool(named),
		[
			(resolution, caller, target)
			for resolution, caller, target in named
			if target not in fallback
		],
	) == (True, [])


def _lower_bound(report: StackReport) -> int:
	match report.bound:
		case Bounded(bytes=depth):
			return depth
		case Unbounded(at_least=at_least):
			return at_least
		case _ as unreachable:
			assert_never(unreachable)


def _depths(
	expansion: Expansion,
	targets_by_caller: Mapping[str, frozenset[str]],
	fallback: frozenset[str],
	*,
	exact: bool,
) -> Mapping[str, int]:
	return {
		report.entry: _lower_bound(report)
		for report in worst_case_depths(
			expand_indirect_calls(
				expansion.in_image_edges, targets_by_caller, fallback, exact=exact
			),
			expansion.frames,
			entry_edges=expansion.in_image_edges,
		)
	}


@pytest.mark.parametrize("name", FIXTURES)
def test_no_entry_is_deeper_with_its_candidates_alone_than_with_the_fallback_too(
	expansions: Mapping[str, Expansion], name: str
) -> None:
	expansion = expansions[name]
	targets_by_caller, fallback = per_caller_candidates(
		expansion.program,
		expansion.resolution.sites,
		expansion.resolution.assignments,
		names=stack_names(expansion.program),
	)
	exact = _depths(expansion, targets_by_caller, fallback, exact=True)
	sound = _depths(expansion, targets_by_caller, fallback, exact=False)
	assert (
		any(edge.callee == INDIRECT_CALLEE for edge in expansion.in_image_edges),
		{
			entry: (exact[entry], sound[entry])
			for entry in exact.keys() & sound.keys()
			if exact[entry] > sound[entry]
		},
	) == (True, {})
