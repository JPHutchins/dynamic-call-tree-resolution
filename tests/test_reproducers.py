# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Soundness reproducers, checked against the targets their runs really call."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from itertools import groupby
from typing import TYPE_CHECKING, assert_never

import pytest
from salix import Struct

from dynamic_call_tree_resolution import (
	Unbounded,
	assignments,
	build_report,
	extract_call_sites,
	load,
)
from dynamic_call_tree_resolution.call_sites import per_caller_candidates
from dynamic_call_tree_resolution.callgraph import load_callgraph
from dynamic_call_tree_resolution.stack_analysis import (
	expand_indirect_calls,
	frame_key,
	worst_case_depths,
)
from dynamic_call_tree_resolution.stack_usage import load_stack_usages
from dynamic_call_tree_resolution.vsa import address_taken
from tests.toolchains import FIXTURES, build_cortex_m3, build_host, run_cortex_m3, run_host

if TYPE_CHECKING:
	import subprocess
	from collections.abc import Iterator, Mapping
	from pathlib import Path

	from dynamic_call_tree_resolution.callgraph import CallEdge

CORTEX_M3_LEVELS = ("-O0", "-O2", "-Os")

ARGUMENTS = ("x", "y")

SEED_OVERFLOW_TARGETS = 65


class Platform(Enum):
	CORTEX_M3 = "cortex-m3"
	HOST = "host"


class Image(Struct):
	"""One reproducer build."""

	source: str
	"""A file under ``fixtures/reproducers``, or a key of ``GENERATED_SOURCES``."""
	platform: Platform
	flags: tuple[str, ...]


class Case(Struct):
	"""One reproducer cell."""

	image: Image
	caller: str


def _seed_overflow_source() -> str:
	return "\n".join(
		(
			'#include "harness.h"',
			"",
			*(
				f'void f{index}(void) {{\n\tobserve("f{index}");\n}}\n'
				for index in range(SEED_OVERFLOW_TARGETS)
			),
			"[[gnu::noipa]] void run(void ( * const fp)(void)) {\n\tfp();\n}\n",
			"int main(int argc, [[maybe_unused]] char * argv[argc + 1]) {",
			'\tobserve("@run");',
			*(f"\trun(f{index});" for index in range(SEED_OVERFLOW_TARGETS)),
			"\treturn 0;",
			"}",
			"",
		)
	)


GENERATED_SOURCES: Mapping[str, str] = {"seed_overflow.c": _seed_overflow_source()}


def _unsound(issues: tuple[int, ...]) -> tuple[pytest.MarkDecorator, ...]:
	return (
		(
			pytest.mark.xfail(
				strict=True,
				raises=AssertionError,
				reason=" ".join(
					f"https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/{issue}"
					for issue in issues
				),
			),
		)
		if issues
		else ()
	)


def _image_id(image: Image) -> str:
	return f"{image.platform.value}:{image.source.removesuffix('.c')}{''.join(image.flags)}"


def _cortex_m3(
	source: str, caller: str, issues_by_level: Mapping[str, tuple[int, ...]]
) -> Iterator[tuple[Case, tuple[int, ...]]]:
	return (
		(
			Case(
				image=Image(source=source, platform=Platform.CORTEX_M3, flags=(level,)),
				caller=caller,
			),
			issues_by_level.get(level, ()),
		)
		for level in CORTEX_M3_LEVELS
	)


def _host(
	source: str, caller: str, flags: tuple[str, ...], issues: tuple[int, ...]
) -> tuple[Case, tuple[int, ...]]:
	return Case(
		image=Image(source=source, platform=Platform.HOST, flags=flags), caller=caller
	), issues


def _everywhere(*issues: int) -> Mapping[str, tuple[int, ...]]:
	return dict.fromkeys(CORTEX_M3_LEVELS, issues)


CANDIDATES = (
	*_cortex_m3("top_argument.c", "run", {}),
	*_cortex_m3("unobserved_callers.c", "run", {}),
	*_cortex_m3("top_stores.c", "call_bss", {}),
	*_cortex_m3("top_stores.c", "call_data", {}),
	*_cortex_m3("top_stores.c", "call_object", {}),
	*_cortex_m3("predicated_store.c", "predicated_store_case", {}),
	*_cortex_m3("call_clobbers.c", "stack_case", {}),
	*_cortex_m3("call_clobbers.c", "global_case", {}),
	*_cortex_m3("seed_overflow.c", "run", {}),
	*_cortex_m3("round_cap.c", "w9", {}),
	*_cortex_m3("jump_table.c", "switch_case", _everywhere(66)),
	*_cortex_m3("writeback_walk.c", "walk", {"-O2": (66,)}),
	*_cortex_m3("predicated_call.c", "predicated_call_case", _everywhere(66)),
	*_cortex_m3("cast_handler.c", "main", {}),
	_host("jump_table.c", "switch_case", ("-O0", "-no-pie", "-fcf-protection=full"), (66,)),
	_host("jump_table.c", "switch_case", ("-O2", "-no-pie", "-fcf-protection=full"), (66,)),
	_host("x86_64_subregister.c", "subregister_case", ("-O2", "-no-pie"), (113,)),
	_host("cast_handler.c", "main", ("-O2", "-no-pie"), ()),
	_host("cast_handler.c", "main", ("-Os", "-no-pie"), ()),
)

STACK_IMAGES = tuple(
	Image(
		source="code_address_callback.c",
		platform=Platform.CORTEX_M3,
		flags=(level, "-fstack-usage", "-fcallgraph-info=su,da", *variant),
	)
	for variant in ((), ("-DPARTIAL",))
	for level in CORTEX_M3_LEVELS
)

OBSERVED_CASES = (
	*(case for case, _ in CANDIDATES),
	*(Case(image=image, caller="main") for image in STACK_IMAGES),
)


IMAGES = tuple(
	sorted(frozenset((*(case.image for case, _ in CANDIDATES), *STACK_IMAGES)), key=_image_id)
)


class Outcome(Struct):
	"""One image, built, run and analyzed."""

	elf: Path
	observations: Mapping[str, frozenset[str]]
	"""Target names the run observed, by caller."""
	sites: tuple[tuple[str, frozenset[str]], ...]
	"""Each call site's caller and dctr's candidate names."""


def _build(image: Image, directory: Path) -> Path:
	source = directory / image.source
	source.write_text(
		GENERATED_SOURCES.get(image.source) or (FIXTURES / "reproducers" / image.source).read_text()
	)
	match image.platform:
		case Platform.CORTEX_M3:
			return build_cortex_m3((source,), directory / "image.elf", "-g", *image.flags)
		case Platform.HOST:
			return build_host((source,), directory / "image.elf", "-g", *image.flags)
		case _ as unreachable:
			assert_never(unreachable)


def _run(platform: Platform, elf: Path) -> subprocess.CompletedProcess[str]:
	match platform:
		case Platform.CORTEX_M3:
			return run_cortex_m3(elf, *ARGUMENTS)
		case Platform.HOST:
			return run_host(elf, *ARGUMENTS)
		case _ as unreachable:
			assert_never(unreachable)


def _observations(result: subprocess.CompletedProcess[str]) -> Mapping[str, frozenset[str]]:
	result.check_returncode()
	lines = result.stdout.splitlines()
	markers = tuple(index for index, line in enumerate(lines) if line.startswith("@"))
	return {
		lines[start].removeprefix("@"): frozenset(lines[start + 1 : end])
		for start, end in zip(markers, (*markers[1:], len(lines)), strict=True)
	}


def _sites(elf: Path) -> tuple[tuple[str, frozenset[str]], ...]:
	program = load(elf)
	return tuple(
		(site.caller, frozenset(candidate.name for candidate in site.candidates))
		for site in build_report(
			program, assignments(program), extract_call_sites(program)
		).call_sites
	)


def _outcome(image: Image, directory: Path) -> Outcome:
	elf = _build(image, directory)
	return Outcome(
		elf=elf, observations=_observations(_run(image.platform, elf)), sites=_sites(elf)
	)


@pytest.fixture(scope="session")
def outcomes(tmp_path_factory: pytest.TempPathFactory) -> Mapping[Image, Outcome]:
	directories = tuple(
		tmp_path_factory.mktemp(image.source.removesuffix(".c")) for image in IMAGES
	)
	with ThreadPoolExecutor() as pool:
		return dict(zip(IMAGES, pool.map(_outcome, IMAGES, directories), strict=True))


@pytest.mark.parametrize(
	"case",
	[pytest.param(case, id=f"{_image_id(case.image)}:{case.caller}") for case in OBSERVED_CASES],
)
def test_run_observes_targets_for_the_caller(case: Case, outcomes: Mapping[Image, Outcome]) -> None:
	assert outcomes[case.image].observations[case.caller]


@pytest.mark.parametrize(
	"case",
	[
		pytest.param(case, id=f"{_image_id(case.image)}:{case.caller}", marks=_unsound(issues))
		for case, issues in CANDIDATES
	],
)
def test_every_observed_target_is_a_candidate_of_each_resolved_site(
	case: Case, outcomes: Mapping[Image, Outcome]
) -> None:
	outcome = outcomes[case.image]
	observed = outcome.observations[case.caller]
	sites = tuple(names for caller, names in outcome.sites if caller == case.caller)
	assert sites
	assert [names for names in sites if names and not observed <= names] == []


def _expanded_call_graph(elf: Path) -> tuple[CallEdge, ...]:
	program = load(elf)
	targets_by_caller, fallback = per_caller_candidates(
		program, extract_call_sites(program), assignments(program)
	)
	return expand_indirect_calls(load_callgraph(elf.parent), targets_by_caller, fallback)


def _caller_key(edge: CallEdge) -> str:
	return frame_key(edge.caller)


def _closure(reached: frozenset[str], callees: Mapping[str, frozenset[str]]) -> frozenset[str]:
	grown = reached | frozenset(
		callee for caller in reached for callee in callees.get(caller, frozenset())
	)
	return reached if grown == reached else _closure(grown, callees)


def _reachable(edges: tuple[CallEdge, ...], entry: str) -> frozenset[str]:
	return _closure(
		frozenset({entry}),
		{
			caller: frozenset(frame_key(edge.callee) for edge in group)
			for caller, group in groupby(sorted(edges, key=_caller_key), key=_caller_key)
		},
	)


@pytest.mark.parametrize(
	"image", [pytest.param(image, id=_image_id(image)) for image in STACK_IMAGES]
)
def test_the_expanded_graph_reaches_every_target_the_run_called(
	image: Image, outcomes: Mapping[Image, Outcome]
) -> None:
	outcome = outcomes[image]
	assert (
		outcome.observations["main"] - _reachable(_expanded_call_graph(outcome.elf), "reset")
		== set()
	)


@pytest.mark.parametrize(
	"image", [pytest.param(image, id=_image_id(image)) for image in STACK_IMAGES]
)
def test_a_bounded_reset_covers_the_frames_on_the_path_the_run_took(
	image: Image, outcomes: Mapping[Image, Outcome]
) -> None:
	elf = outcomes[image].elf
	frames = load_stack_usages(elf.parent)
	(reset,) = (
		report
		for report in worst_case_depths(
			_expanded_call_graph(elf), frames, entry_edges=load_callgraph(elf.parent)
		)
		if report.entry == "reset"
	)
	bytes_by_function = {frame.function: frame.bytes for frame in frames}
	assert isinstance(reset.bound, Unbounded) or reset.bound.bytes >= sum(
		bytes_by_function[function] for function in ("reset", "main", "deep")
	)


@pytest.mark.parametrize("image", [pytest.param(image, id=_image_id(image)) for image in IMAGES])
def test_every_function_a_site_candidate_names_is_address_taken(
	image: Image, outcomes: Mapping[Image, Outcome]
) -> None:
	program = load(outcomes[image].elf)
	taken = address_taken(program)
	assert [
		program.functions[address].name
		for site in extract_call_sites(program)
		for address in site.candidates
		if address in program.functions and address not in taken
	] == []
