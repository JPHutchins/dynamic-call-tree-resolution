# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Invariants every fixture's analysis holds, whatever its numbers (#75, item 2)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from functools import partial
from typing import TYPE_CHECKING, Final, assert_never

import pytest
from salix import Struct

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
	NO_CALLEE,
	expand_indirect_calls,
	frame_key,
	worst_case_depths,
)
from dynamic_call_tree_resolution.vsa.abi import normalized

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path

	from dynamic_call_tree_resolution.call_sites import CallerCandidates
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
	FIXTURES,
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


class _Narrowings(Struct):
	"""The narrowings one parametrized run turns on."""

	by_signature: bool
	by_field: tuple[NarrowedSpan, ...]


def _narrowed(fixtures: Path, name: str, expansion: Expansion, narrowing: Narrowing) -> _Narrowings:
	match narrowing:
		case Narrowing.NONE:
			return _Narrowings(False, ())
		case Narrowing.FIELD:
			return _Narrowings(False, _narrowed_by_field(fixtures, name, expansion))
		case Narrowing.SIGNATURE:
			return _Narrowings(True, ())
		case _ as unreachable:
			assert_never(unreachable)


class _Named(Struct):
	"""A target a resolution names for one caller."""

	resolution: str
	"""``global``, or the thread whose own analysis named it."""
	caller: str
	target: str


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
	narrowings = _narrowed(zephyr_fixtures, name, expansion, narrowing)
	candidates = per_caller_candidates(
		expansion.program,
		expansion.resolution.sites,
		expansion.resolution.assignments,
		narrow_by_signature=narrowings.by_signature,
		narrowed_by_field=narrowings.by_field,
		names=names,
	)
	named = [
		_Named(resolution=resolution, caller=caller, target=target)
		for resolution, by_caller in {
			"global": candidates.targets_by_caller,
			**{
				thread: own_targets(
					expansion.program,
					sites,
					expansion.resolution.assignments,
					names,
					narrowings.by_field,
					narrow_by_signature=narrowings.by_signature,
				).targets_by_caller
				for thread, sites in expansion.resolution.threads.items()
			},
		}.items()
		for caller, targets in by_caller.items()
		for target in targets - {INDIRECT_CALLEE, NO_CALLEE}
	]
	assert (
		bool(named),
		[found for found in named if found.target not in candidates.fallback],
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
	candidates: CallerCandidates,
	*,
	exact: bool,
) -> Mapping[str, int]:
	return {
		report.entry: _lower_bound(report)
		for report in worst_case_depths(
			expand_indirect_calls(
				expansion.in_image_edges,
				candidates.targets_by_caller,
				candidates.fallback,
				exact=exact,
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
	candidates = per_caller_candidates(
		expansion.program,
		expansion.resolution.sites,
		expansion.resolution.assignments,
		names=stack_names(expansion.program),
	)
	exact = _depths(expansion, candidates, exact=True)
	sound = _depths(expansion, candidates, exact=False)
	assert (
		any(edge.callee == INDIRECT_CALLEE for edge in expansion.in_image_edges),
		{
			entry: (exact[entry], sound[entry])
			for entry in exact.keys() & sound.keys()
			if exact[entry] > sound[entry]
		},
	) == (True, {})
