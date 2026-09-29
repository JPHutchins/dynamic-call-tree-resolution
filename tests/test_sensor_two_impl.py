# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The sensor-two-impl fixture pins the per-thread resolution proof.

``tests/fixtures/sensor-two-impl-app`` builds two identical threads,
each bound to a different existing sensor driver (ADT7420 and BMI160)
on one emulated I2C bus; the committed artifacts (produced by the
``sensor_two_impl`` task) let the tests here replay the analysis and
pin each thread's dispatch to its own impl.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec
import pytest
from salix import Struct

from dynamic_call_tree_resolution import (
	AnalysisReport,
	assignments,
	extract_call_sites,
	load,
)
from dynamic_call_tree_resolution.call_sites import per_caller_candidates
from dynamic_call_tree_resolution.callgraph import load_callgraph
from dynamic_call_tree_resolution.stack_analysis import (
	expand_indirect_calls,
	worst_case_depths,
)
from dynamic_call_tree_resolution.stack_usage import load_stack_usages

if TYPE_CHECKING:
	from collections.abc import Mapping

	from dynamic_call_tree_resolution.callgraph import CallEdge
	from dynamic_call_tree_resolution.stack_analysis import StackReport
	from dynamic_call_tree_resolution.stack_usage import StackUsage

ARTIFACTS = Path(__file__).parent / "fixtures" / "sensor-two-impl"
EXECUTABLE = ARTIFACTS / "zephyr" / "zephyr.elf"


def _dctr(*arguments: str) -> str:
	executable = Path(sys.executable).with_name("dctr")
	result = subprocess.run(
		[str(executable), *arguments],
		check=True,
		capture_output=True,
		text=True,
	)
	return result.stdout


def test_analyze_resolves_each_threads_dispatch_to_its_own_impl() -> None:
	report = msgspec.json.decode(_dctr("analyze", "--json", str(EXECUTABLE)), type=AnalysisReport)
	sites_by_caller: dict[str, set[str]] = {}
	for site in report.call_sites:
		sites_by_caller.setdefault(site.caller, set()).update(
			candidate.name for candidate in site.candidates
		)
	assert sites_by_caller["thermal_thread"] == {"adt7420_sample_fetch", "adt7420_channel_get"}
	assert sites_by_caller["motion_thread"] == {"bmi160_sample_fetch", "bmi160_channel_get"}


class _Resolution(Struct):
	"""The committed call graph and frames, with dctr's candidates from the committed ELF."""

	edges: tuple[CallEdge, ...]
	frames: tuple[StackUsage, ...]
	targets_by_caller: Mapping[str, frozenset[str]]
	fallback: frozenset[str]


@pytest.fixture(scope="module")
def resolution() -> _Resolution:
	program = load(EXECUTABLE)
	targets_by_caller, fallback = per_caller_candidates(
		program, extract_call_sites(program), assignments(program)
	)
	return _Resolution(
		edges=load_callgraph(ARTIFACTS),
		frames=load_stack_usages(ARTIFACTS),
		targets_by_caller=targets_by_caller,
		fallback=fallback,
	)


def _depths(resolution: _Resolution, *, exact: bool) -> Mapping[str, StackReport]:
	return {
		report.entry: report
		for report in worst_case_depths(
			expand_indirect_calls(
				resolution.edges, resolution.targets_by_caller, resolution.fallback, exact=exact
			),
			resolution.frames,
			entry_edges=resolution.edges,
		)
	}


@pytest.fixture(scope="module")
def exact_depths(resolution: _Resolution) -> Mapping[str, StackReport]:
	return _depths(resolution, exact=True)


@pytest.fixture(scope="module")
def sound_depths(resolution: _Resolution) -> Mapping[str, StackReport]:
	return _depths(resolution, exact=False)


def test_exact_expansion_gives_each_thread_its_impl_depth(
	exact_depths: Mapping[str, StackReport],
) -> None:
	assert exact_depths["thermal_thread"].depth == 72
	assert exact_depths["thermal_thread"].recursive is False
	assert exact_depths["motion_thread"].depth == 648
	assert exact_depths["motion_thread"].recursive is True


@pytest.mark.xfail(
	strict=True,
	raises=AssertionError,
	reason="https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/58",
)
def test_no_entry_is_deeper_exact_than_sound(
	exact_depths: Mapping[str, StackReport], sound_depths: Mapping[str, StackReport]
) -> None:
	assert {
		entry: (exact_depths[entry].depth, sound_depths[entry].depth)
		for entry in exact_depths.keys() & sound_depths.keys()
		if exact_depths[entry].depth > sound_depths[entry].depth
	} == {}
