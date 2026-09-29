# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Soundness reproducers, checked against the targets their runs really call.

Each reproducer's ``main`` observes ``@caller`` before calling a function
under test, and each target observes its own name.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import TYPE_CHECKING

import pytest
from salix import Struct

from dynamic_call_tree_resolution import assignments, build_report, extract_call_sites, load
from dynamic_call_tree_resolution.call_sites import per_caller_candidates
from dynamic_call_tree_resolution.callgraph import load_callgraph
from dynamic_call_tree_resolution.stack_analysis import expand_indirect_calls, worst_case_depths
from dynamic_call_tree_resolution.stack_usage import load_stack_usages
from tests.toolchains import FIXTURES, build_cortex_m3, build_host, run_cortex_m3, run_host

if TYPE_CHECKING:
	import subprocess
	from collections.abc import Iterator, Mapping
	from pathlib import Path

CORTEX_M3_LEVELS = ("-O0", "-O2", "-Os")

ARGUMENTS = ("x", "y")

SEED_OVERFLOW_TARGETS = 65


class Platform(Enum):
	CORTEX_M3 = "cortex-m3"
	HOST = "host"


class Image(Struct):
	"""One reproducer build: a source under ``fixtures/reproducers`` or a generated one."""

	source: str
	platform: Platform
	flags: tuple[str, ...]


class Case(Struct):
	"""One caller under test in one image."""

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
	*_cortex_m3("top_argument.c", "run", _everywhere(61)),
	*_cortex_m3("unobserved_callers.c", "run", {"-O0": (61,), "-O2": (61, 66), "-Os": (61, 66)}),
	*_cortex_m3("top_stores.c", "call_bss", _everywhere(61)),
	*_cortex_m3("top_stores.c", "call_data", _everywhere(61)),
	*_cortex_m3("top_stores.c", "call_object", _everywhere(61)),
	*_cortex_m3("predicated_store.c", "predicated_store_case", {"-O2": (61,), "-Os": (61,)}),
	*_cortex_m3("call_clobbers.c", "stack_case", {"-O0": (61,), "-Os": (61,)}),
	*_cortex_m3("call_clobbers.c", "global_case", _everywhere(61)),
	*_cortex_m3("seed_overflow.c", "run", _everywhere(61)),
	*_cortex_m3("round_cap.c", "w9", {"-O0": (61,), "-O2": (61, 66), "-Os": (61, 66)}),
	*_cortex_m3("jump_table.c", "switch_case", _everywhere(66)),
	*_cortex_m3("writeback_walk.c", "walk", {"-O2": (66,)}),
	*_cortex_m3("predicated_call.c", "predicated_call_case", _everywhere(66)),
	_host("jump_table.c", "switch_case", ("-O0", "-no-pie", "-fcf-protection=full"), (66,)),
	_host("jump_table.c", "switch_case", ("-O2", "-no-pie", "-fcf-protection=full"), (66,)),
	_host("x86_64_subregister.c", "subregister_case", ("-O2", "-no-pie"), (61,)),
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


IMAGES = tuple(frozenset((*(case.image for case, _ in CANDIDATES), *STACK_IMAGES)))


class Outcome(Struct):
	"""One image: its path, what its run observed per caller, and dctr's sites per caller."""

	elf: Path
	observations: Mapping[str, frozenset[str]]
	sites: tuple[tuple[str, frozenset[str]], ...]


def _build(image: Image, directory: Path) -> Path:
	source = directory / image.source
	source.write_text(
		GENERATED_SOURCES.get(image.source) or (FIXTURES / "reproducers" / image.source).read_text()
	)
	match image.platform:
		case Platform.CORTEX_M3:
			return build_cortex_m3((source,), directory / "image.elf", "-g", *image.flags)
		case Platform.HOST:  # pragma: no branch
			return build_host((source,), directory / "image.elf", "-g", *image.flags)


def _run(platform: Platform, elf: Path) -> subprocess.CompletedProcess[str]:
	match platform:
		case Platform.CORTEX_M3:
			return run_cortex_m3(elf, *ARGUMENTS)
		case Platform.HOST:  # pragma: no branch
			return run_host(elf, *ARGUMENTS)


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
	"""Every image built, run and analyzed concurrently, each in its own directory."""
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


@pytest.mark.parametrize(
	"image",
	[pytest.param(image, id=_image_id(image), marks=_unsound((62,))) for image in STACK_IMAGES],
)
def test_the_reset_bound_covers_the_frames_on_the_path_the_run_took(
	image: Image, outcomes: Mapping[Image, Outcome]
) -> None:
	elf = outcomes[image].elf
	program = load(elf)
	edges = load_callgraph(elf.parent)
	frames = load_stack_usages(elf.parent)
	targets_by_caller, fallback = per_caller_candidates(
		program, extract_call_sites(program), assignments(program)
	)
	(reset,) = (
		report
		for report in worst_case_depths(
			expand_indirect_calls(edges, targets_by_caller, fallback), frames, entry_edges=edges
		)
		if report.entry == "reset"
	)
	bytes_by_function = {frame.function: frame.bytes for frame in frames}
	assert reset.depth >= sum(bytes_by_function[function] for function in ("reset", "main", "deep"))
