# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The sensor-two-impl fixture: two threads on one emulated I2C bus, one sensor driver each."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

import msgspec
import pytest
from salix import Struct

from dynamic_call_tree_resolution import (
	AnalysisReport,
	Bounded,
	StackReport,
	Unbounded,
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
	from dynamic_call_tree_resolution.stack_usage import StackUsage

pytestmark = pytest.mark.image

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


def _lower_bound(report: StackReport) -> int:
	match report.bound:
		case Bounded(bytes=depth):
			return depth
		case Unbounded(at_least=at_least):
			return at_least
		case _ as unreachable:
			assert_never(unreachable)


def test_exact_expansion_joins_both_threads_in_the_i2c_emulator_cycle(
	exact_depths: Mapping[str, StackReport],
) -> None:
	thermal = exact_depths["thermal_thread"].bound
	motion = exact_depths["motion_thread"].bound
	assert isinstance(thermal, Unbounded)
	assert isinstance(motion, Unbounded)
	assert (thermal.at_least, motion.at_least) == (848, 824)
	assert thermal.recursion == motion.recursion
	assert {"thermal_thread", "motion_thread", "i2c_write_read", "i2c_emul_transfer"} <= (
		thermal.recursion
	)


@pytest.mark.xfail(
	strict=True,
	raises=AssertionError,
	reason="https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/58",
)
def test_no_entry_is_deeper_exact_than_sound(
	exact_depths: Mapping[str, StackReport], sound_depths: Mapping[str, StackReport]
) -> None:
	assert {
		entry: (_lower_bound(exact_depths[entry]), _lower_bound(sound_depths[entry]))
		for entry in exact_depths.keys() & sound_depths.keys()
		if _lower_bound(exact_depths[entry]) > _lower_bound(sound_depths[entry])
	} == {}
