# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Soundness reproducers, checked against the targets their runs really call."""

from __future__ import annotations

import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from itertools import groupby
from typing import TYPE_CHECKING, Final, assert_never

import pytest
from salix import Struct

from dynamic_call_tree_resolution import (
	Unbounded,
	build_report,
	extract_call_sites,
	load,
	resolve,
)
from dynamic_call_tree_resolution.call_sites import per_caller_candidates
from dynamic_call_tree_resolution.callgraph import load_callgraph
from dynamic_call_tree_resolution.model import Address, RtosModel, ThreadRoot, render_path
from dynamic_call_tree_resolution.points_to import assignments, pointer_at
from dynamic_call_tree_resolution.stack_analysis import (
	expand_indirect_calls,
	frame_key,
	worst_case_depths,
)
from dynamic_call_tree_resolution.stack_usage import load_stack_usages
from dynamic_call_tree_resolution.vsa import address_taken
from tests.toolchains import (
	FIXTURES,
	build_cortex_a15,
	build_cortex_m3,
	build_host,
	run_cortex_a15,
	run_cortex_m3,
	run_host,
)

if TYPE_CHECKING:
	from collections.abc import Iterator, Mapping
	from pathlib import Path

	from dynamic_call_tree_resolution.call_sites import ProgramResolution
	from dynamic_call_tree_resolution.callgraph import CallEdge
	from dynamic_call_tree_resolution.model import CallSite, Program

ARM_LEVELS = ("-O0", "-O2", "-Os")

ARGUMENTS = ("x", "y")

SEED_OVERFLOW_TARGETS = 65


class Platform(Enum):
	CORTEX_M3 = "cortex-m3"
	CORTEX_A15 = "cortex-a15"
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


def _arm(
	platform: Platform, source: str, caller: str, issues_by_level: Mapping[str, tuple[int, ...]]
) -> Iterator[tuple[Case, tuple[int, ...]]]:
	return (
		(
			Case(image=Image(source=source, platform=platform, flags=(level,)), caller=caller),
			issues_by_level.get(level, ()),
		)
		for level in ARM_LEVELS
	)


ARM_CASES: tuple[tuple[str, str], ...] = (
	("top_argument.c", "run"),
	("unobserved_callers.c", "run"),
	("top_stores.c", "call_bss"),
	("top_stores.c", "call_data"),
	("top_stores.c", "call_object"),
	("predicated_store.c", "predicated_store_case"),
	("call_clobbers.c", "stack_case"),
	("call_clobbers.c", "global_case"),
	("seed_overflow.c", "run"),
	("round_cap.c", "w9"),
	("jump_table.c", "switch_case"),
	("writeback_walk.c", "walk"),
	("cast_handler.c", "main"),
)


def _host(
	source: str, caller: str, flags: tuple[str, ...], issues: tuple[int, ...]
) -> tuple[Case, tuple[int, ...]]:
	return Case(
		image=Image(source=source, platform=Platform.HOST, flags=flags), caller=caller
	), issues


CANDIDATES = (
	*(
		cell
		for source, caller in (*ARM_CASES, ("predicated_call.c", "predicated_call_case"))
		for cell in _arm(Platform.CORTEX_M3, source, caller, {})
	),
	*(
		cell
		for source, caller in (*ARM_CASES, ("predicated_call.c", "predicated_call_case"))
		for cell in _arm(Platform.CORTEX_A15, source, caller, {})
	),
	_host("jump_table.c", "switch_case", ("-O0", "-no-pie", "-fcf-protection=full"), (113,)),
	_host("jump_table.c", "switch_case", ("-O2", "-no-pie", "-fcf-protection=full"), (113,)),
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
	for level in ARM_LEVELS
)

THREAD_RECORD_IMAGES = tuple(
	Image(source="thread_record.c", platform=platform, flags=flags)
	for platform, flags in (
		*((Platform.CORTEX_M3, (level,)) for level in ARM_LEVELS),
		(Platform.CORTEX_A15, ("-O2",)),
		(Platform.HOST, ("-O2", "-no-pie")),
	)
)

ESCAPED_RECORD_IMAGES = tuple(
	Image(source=image.source, platform=image.platform, flags=(*image.flags, "-DESCAPED"))
	for image in THREAD_RECORD_IMAGES
)

SHARED_ENTRY_IMAGES = tuple(
	Image(source="shared_entry.c", platform=image.platform, flags=image.flags)
	for image in THREAD_RECORD_IMAGES
)

OBSERVED_CASES = (
	*(case for case, _ in CANDIDATES),
	*(Case(image=image, caller="main") for image in STACK_IMAGES),
	*(
		Case(image=image, caller="worker")
		for image in (*THREAD_RECORD_IMAGES, *ESCAPED_RECORD_IMAGES)
	),
	*(
		Case(image=image, caller=thread)
		for image in SHARED_ENTRY_IMAGES
		for thread in ("thread_a", "thread_b")
	),
)


IMAGES = tuple(
	sorted(
		frozenset(
			(
				*(case.image for case, _ in CANDIDATES),
				*STACK_IMAGES,
				*THREAD_RECORD_IMAGES,
				*ESCAPED_RECORD_IMAGES,
				*SHARED_ENTRY_IMAGES,
			)
		),
		key=_image_id,
	)
)

A15_IMAGES = tuple(image for image in IMAGES if image.platform is Platform.CORTEX_A15)

REGISTER_INDIRECT_BRANCH: Final = re.compile(
	r"\t(?:blx|bx)(?:eq|ne|cs|cc|hs|lo|mi|pl|vs|vc|hi|ls|ge|lt|gt|le)?(?:\.[nw])?"
	r"\t(?!lr\b)(?:r\d+|sb|sl|fp|ip)\b"
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
		case Platform.CORTEX_A15:
			return build_cortex_a15((source,), directory / "image.elf", "-g", *image.flags)
		case Platform.HOST:
			return build_host((source,), directory / "image.elf", "-g", *image.flags)
		case _ as unreachable:
			assert_never(unreachable)


def _run(platform: Platform, elf: Path) -> subprocess.CompletedProcess[str]:
	match platform:
		case Platform.CORTEX_M3:
			return run_cortex_m3(elf, *ARGUMENTS)
		case Platform.CORTEX_A15:
			return run_cortex_a15(elf, *ARGUMENTS)
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
	resolution = resolve(program)
	return tuple(
		(site.caller, frozenset(candidate.name for candidate in site.candidates))
		for site in build_report(program, resolution.assignments, resolution.sites).call_sites
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
def test_every_observed_target_is_a_candidate_of_the_callers_sites(
	case: Case, outcomes: Mapping[Image, Outcome]
) -> None:
	outcome = outcomes[case.image]
	observed = outcome.observations[case.caller]
	sites = tuple(names for caller, names in outcome.sites if caller == case.caller)
	assert sites
	union = frozenset[str]().union(*sites)
	assert not union or observed <= union


def _expanded_call_graph(elf: Path) -> tuple[CallEdge, ...]:
	program = load(elf)
	resolution = resolve(program)
	targets_by_caller, fallback = per_caller_candidates(
		program, resolution.sites, resolution.assignments
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


@pytest.mark.parametrize(
	"image",
	[
		pytest.param(image, id=_image_id(image))
		for image in IMAGES
		if image.source == "jump_table.c" and image.platform is not Platform.HOST
	],
)
def test_switch_case_sites_name_every_case(image: Image, outcomes: Mapping[Image, Outcome]) -> None:
	assert frozenset[str]().union(
		*(names for caller, names in outcomes[image].sites if caller == "switch_case")
	) == {"a", "b", "c", "d", "e"}


@pytest.mark.parametrize(
	"image",
	[pytest.param(image, id=_image_id(image)) for image in A15_IMAGES],
)
def test_a32_sites_are_objdumps_register_indirect_branches(
	image: Image, outcomes: Mapping[Image, Outcome]
) -> None:
	listing = subprocess.run(
		["arm-none-eabi-objdump", "-d", str(outcomes[image].elf)],
		check=True,
		capture_output=True,
		text=True,
	).stdout
	assert len(outcomes[image].sites) == sum(1 for _ in REGISTER_INDIRECT_BRANCH.finditer(listing))


def _record_thread(program: Program, name: str, index: int) -> ThreadRoot:
	(records,) = (item for item in program.objects.values() if item.name == "records")
	layout = program.layouts["struct thread_record"]
	offsets = {
		member.name: records.address + index * layout.size + member.offset
		for member in layout.members
	}
	return ThreadRoot(
		name=name,
		entry=Address(pointer_at(program, Address(offsets["entry"])) or 0),
		entry_slot=Address(offsets["entry"]),
		arguments=tuple(
			Address(pointer_at(program, Address(offsets[member])) or 0)
			for member in ("p1", "p2", "p3")
		),
	)


def _worker_targets(
	program: Program, resolution: ProgramResolution, sites: tuple[CallSite, ...]
) -> frozenset[str]:
	return frozenset(
		candidate.name
		for site in build_report(program, resolution.assignments, sites).call_sites
		if site.caller == "worker"
		for candidate in site.candidates
	)


def _with_record_thread(program: Program) -> ProgramResolution:
	return resolve(
		program,
		RtosModel(
			name="test",
			evidence=(),
			threads=(_record_thread(program, "record", 0),),
			trampoline=None,
		),
	)


@pytest.mark.parametrize(
	"image", [pytest.param(image, id=_image_id(image)) for image in THREAD_RECORD_IMAGES]
)
def test_a_seeded_entry_resolves_to_exactly_what_its_record_passes(
	image: Image, outcomes: Mapping[Image, Outcome]
) -> None:
	program = load(outcomes[image].elf)
	seeded = _with_record_thread(program)
	unseeded = resolve(program)
	assert (
		_worker_targets(program, unseeded, unseeded.sites),
		seeded.seeded,
		_worker_targets(program, seeded, seeded.sites),
	) == (frozenset(), frozenset({"record"}), outcomes[image].observations["worker"])


@pytest.mark.parametrize(
	"image", [pytest.param(image, id=_image_id(image)) for image in ESCAPED_RECORD_IMAGES]
)
def test_an_entry_whose_address_is_also_stored_elsewhere_is_not_seeded(
	image: Image, outcomes: Mapping[Image, Outcome]
) -> None:
	program = load(outcomes[image].elf)
	seeded = _with_record_thread(program)
	assert (seeded.seeded, _worker_targets(program, seeded, seeded.sites)) == (
		frozenset(),
		frozenset(),
	)


@pytest.mark.parametrize(
	"image",
	[
		pytest.param(image, id=_image_id(image))
		for image in THREAD_RECORD_IMAGES
		if image.platform is Platform.CORTEX_M3
	],
)
def test_a_null_record_argument_leads_to_no_object_at_address_zero(
	image: Image, outcomes: Mapping[Image, Outcome]
) -> None:
	assert sorted(
		render_path(assignment.path)
		for assignment in assignments(load(outcomes[image].elf))
		if assignment.path[0] == "records"
	) == ["records.[0].entry", "records.[0].p1.operations.run"]


@pytest.mark.parametrize(
	"image", [pytest.param(image, id=_image_id(image)) for image in SHARED_ENTRY_IMAGES]
)
def test_each_thread_of_a_shared_entry_resolves_to_what_its_own_record_passes(
	image: Image, outcomes: Mapping[Image, Outcome]
) -> None:
	program = load(outcomes[image].elf)
	resolution = resolve(
		program,
		RtosModel(
			name="test",
			evidence=(),
			threads=(
				_record_thread(program, "thread_a", 0),
				_record_thread(program, "thread_b", 1),
			),
			trampoline="thread_entry",
		),
	)
	assert (
		_worker_targets(program, resolution, resolution.sites),
		{
			name: _worker_targets(program, resolution, thread.sites)
			for name, thread in resolution.threads.items()
		},
	) == (
		outcomes[image].observations["thread_a"] | outcomes[image].observations["thread_b"],
		{name: outcomes[image].observations[name] for name in ("thread_a", "thread_b")},
	)
