# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Image membership from the final link's map and its gc-sections listing (#154)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import pytest

from dynamic_call_tree_resolution import AnalysisSummary
from dynamic_call_tree_resolution.callgraph import callgraph_files, load_callgraph
from dynamic_call_tree_resolution.linker import (
	Linkage,
	LinkerRecords,
	Membership,
	held_artifacts,
	libcall_linkage,
	linkage,
	linker_records,
)
from dynamic_call_tree_resolution.stack_analysis import frame_key
from dynamic_call_tree_resolution.stack_usage import load_stack_usages, stack_usage_files
from tests.dctr import dctr

if TYPE_CHECKING:
	from pathlib import Path


@pytest.mark.image
@pytest.mark.parametrize(
	("artifact", "function", "linked"),
	[
		pytest.param(
			"sys_clock_init.c.ci",
			"sys_clock_set_timeout",
			Linkage.DISCARDED,
			id="the weak sys_clock_set_timeout",
		),
		pytest.param(
			"cortex_m_systick.c.ci",
			"sys_clock_set_timeout",
			Linkage.IN_IMAGE,
			id="the strong sys_clock_set_timeout",
		),
		pytest.param(
			"msg_q.c.ci",
			"unpend_thread_no_timeout",
			Linkage.NEVER_LINKED,
			id="msg_q's static unpend_thread_no_timeout",
		),
		pytest.param(
			"queue.c.ci",
			"unpend_thread_no_timeout",
			Linkage.NEVER_LINKED,
			id="queue's static unpend_thread_no_timeout",
		),
		pytest.param(
			"sched.c.ci",
			"unpend_thread_no_timeout",
			Linkage.IN_IMAGE,
			id="sched's static unpend_thread_no_timeout",
		),
		pytest.param("main_weak.c.ci", "main", Linkage.NEVER_LINKED, id="the weak main"),
		pytest.param(
			"offsets.c.ci",
			"_OffsetAbsSyms",
			Linkage.IN_IMAGE,
			id="an object the map loads through a ./ path",
		),
		pytest.param(
			"heap.c.su",
			"inplace_realloc.isra",
			Linkage.DISCARDED,
			id="a clone .su names without its number",
		),
	],
)
def test_the_final_link_decides_each_copy_of_a_function_by_its_object(
	zephyr_fixtures: Path, artifact: str, function: str, linked: Linkage
) -> None:
	build_directory = zephyr_fixtures / "sensor-two-impl"
	records = linker_records(build_directory / "zephyr" / "zephyr.elf")
	assert records is not None
	(path,) = build_directory.glob(f"**/{artifact}")
	assert linkage(records, build_directory, path, function) is linked


@pytest.mark.image
def test_a_never_linked_objects_edges_and_frames_are_left_out_of_the_graph(
	zephyr_fixtures: Path,
) -> None:
	build_directory = zephyr_fixtures / "sensor-two-impl"
	records = linker_records(build_directory / "zephyr" / "zephyr.elf")
	assert records is not None
	held = held_artifacts(
		records,
		build_directory,
		callgraph_files(build_directory),
		stack_usage_files(build_directory),
	)
	assert (
		"z_impl_k_msgq_get" in {frame_key(edge.caller) for edge in load_callgraph(build_directory)},
		"z_impl_k_msgq_get"
		in {frame_key(edge.caller) for _, edges in held.callgraphs for edge in edges},
		"z_impl_k_msgq_get"
		in {frame_key(usage.function) for usage in load_stack_usages(build_directory)},
		"z_impl_k_msgq_get"
		in {frame_key(usage.function) for _, usages in held.usages for usage in usages},
		"z_impl_k_msgq_get" in held.dropped[Linkage.NEVER_LINKED],
	) == (True, False, True, False, True)


def test_a_libcall_is_in_the_image_if_its_member_kept_code_or_the_image_defines_it(
	tmp_path: Path, fixture_elfs: dict[str, Path]
) -> None:
	elf = tmp_path / "zephyr.elf"
	elf.write_bytes(fixture_elfs["nopie"].read_bytes())
	(tmp_path / "zephyr_final.map").write_text(
		"Archive member included to satisfy reference by file (symbol)\n\n"
		"/sdk/libgcc.a(_aeabi_uldivmod.o)\n"
		"                              zephyr/libzephyr.a(clock.c.obj) (__aeabi_uldivmod)\n"
		"/sdk/libgcc.a(_aeabi_ldivmod.o)\n"
		"                              zephyr/libzephyr.a(timeutil.c.obj) (__aeabi_ldivmod)\n"
		"\nDiscarded input sections\n\n"
		"\nLinker script and memory map\n\n"
		" .text          0x00000000000000f0       0xa0 /sdk/libgcc.a(_aeabi_ldivmod.o)\n"
	)
	(tmp_path / "gc-sections.txt").write_text(
		"removing unused section '.text' in file '/sdk/libgcc.a(_aeabi_uldivmod.o)'\n"
	)
	records = linker_records(elf)
	assert records is not None
	assert {
		name: libcall_linkage(records, name)
		for name in ("__aeabi_uldivmod", "__aeabi_ldivmod", "__aeabi_d2f", "main")
	} == {
		"__aeabi_uldivmod": Linkage.DISCARDED,
		"__aeabi_ldivmod": Linkage.IN_IMAGE,
		"__aeabi_d2f": Linkage.NEVER_LINKED,
		"main": Linkage.IN_IMAGE,
	}


@pytest.mark.image
def test_a_kept_functions_call_to_a_libcall_the_link_dropped_leaves_the_graph(
	zephyr_fixtures: Path,
) -> None:
	build_directory = zephyr_fixtures / "sensor-threads"
	records = linker_records(build_directory / "zephyr" / "zephyr.elf")
	assert records is not None
	held = held_artifacts(
		records,
		build_directory,
		callgraph_files(build_directory),
		stack_usage_files(build_directory),
	)
	callers = {
		callee: sorted(
			{
				frame_key(edge.caller)
				for _, edges in held.callgraphs
				for edge in edges
				if edge.callee == callee
			}
		)
		for callee in ("__aeabi_uldivmod", "__aeabi_ldivmod")
	}
	assert (
		held.phantom_libcalls,
		callers["__aeabi_uldivmod"],
		"bmi160_attr_get" in callers["__aeabi_ldivmod"],
	) == (
		frozenset({"__aeabi_uldivmod"}),
		[],
		True,
	)


def test_an_artifact_outside_a_cmake_target_is_kept(tmp_path: Path) -> None:
	assert (
		linkage(
			LinkerRecords(linked=frozenset(), sections={}),
			tmp_path,
			tmp_path / "loose.c.ci",
			"anything",
		)
		is Linkage.IN_IMAGE
	)


def test_an_image_without_its_map_has_no_linker_records(
	fixture_elfs: dict[str, Path],
) -> None:
	assert linker_records(fixture_elfs["nopie"]) is None


@pytest.mark.image
def test_summary_splits_the_entries_the_linker_left_out(zephyr_fixtures: Path) -> None:
	summary = msgspec.json.decode(
		dctr(
			"summary",
			str(zephyr_fixtures / "sensor-two-impl"),
			str(zephyr_fixtures / "sensor-two-impl" / "zephyr" / "zephyr.elf"),
		),
		type=AnalysisSummary,
	)
	assert (
		summary.membership,
		summary.indirect_call_sites,
		summary.discarded_entry_points,
		summary.never_linked_entry_points,
	) == (Membership.LINKER, 26, 214, 122)
