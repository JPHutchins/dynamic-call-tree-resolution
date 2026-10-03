# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Joining ``.ci`` nodes and ``.su`` records to the image's functions (#155)."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

import pytest

from dynamic_call_tree_resolution import Address, load
from dynamic_call_tree_resolution.callgraph import CallEdge, callgraph_files, callgraph_locations
from dynamic_call_tree_resolution.identity import (
	Identities,
	canonical_edges,
	canonical_frames,
	identities,
	stack_names,
)
from dynamic_call_tree_resolution.model import Declaration, Machine, SourceLocation
from dynamic_call_tree_resolution.stack_analysis import frame_key
from dynamic_call_tree_resolution.stack_usage import stack_usage_files, stack_usage_locations
from tests.programs import build_program

if TYPE_CHECKING:
	from pathlib import Path

	from dynamic_call_tree_resolution.model import Program


def test_a_name_joins_by_declaration_then_by_symbol_then_by_its_copies_and_otherwise_stays() -> (
	None
):
	known = Identities(
		by_declaration={
			Declaration(
				unit="a.c", location=SourceLocation(file="a.c", line=3, column=12)
			): Address(0x10),
		},
		by_symbol={
			"z_alias": frozenset({Address(0x10)}),
			"shared": frozenset({Address(0x20), Address(0x30)}),
		},
		names={
			Address(0x10): "z_alias",
			Address(0x20): "shared@a.c",
			Address(0x30): "shared@b.c",
		},
		copies={"shared": ("shared@a.c", "shared@b.c")},
	)
	assert canonical_edges(
		known,
		"a.c",
		{"a.c:alias": SourceLocation(file="a.c", line=3, column=12)},
		(
			CallEdge(caller="a.c:alias", callee="shared"),
			CallEdge(caller="/src/c.c:shared", callee="prebuilt"),
		),
	) == (
		CallEdge(caller="z_alias", callee="shared@a.c"),
		CallEdge(caller="z_alias", callee="shared@b.c"),
		CallEdge(caller="shared@a.c", callee="prebuilt"),
		CallEdge(caller="shared@b.c", callee="prebuilt"),
	)


@pytest.mark.image
def test_load_reads_each_functions_declaration_and_every_symbol_at_its_address(
	zephyr_fixtures: Path,
) -> None:
	program = load(zephyr_fixtures / "sensor-two-impl" / "zephyr" / "zephyr.elf")
	assert (
		[
			(PurePosixPath(declaration.unit).parts[-7:], declaration.location)
			for address in sorted(program.symbol_addresses["bus_fault.isra.0"])
			for declaration in (program.declarations[address],)
		],
		sorted(
			PurePosixPath(program.declarations[address].unit).name
			for address in program.symbol_addresses["uart_stellaris_init"]
		),
		program.symbol_addresses["z_reschedule"] == program.symbol_addresses["reschedule"],
		program.symbol_addresses["__l_vfprintf"].isdisjoint(program.declarations),
	) == (
		[
			(
				("testbeds", "zephyr", "arch", "arm", "core", "cortex_m", "fault.c"),
				SourceLocation(file="fault.c", line=345, column=12),
			)
		],
		["soc_config.c", "uart_stellaris.c"],
		True,
		True,
	)


def _joins(program: Program, build_directory: Path) -> Counter[Address]:
	known = identities(program, stack_names(program))
	return Counter(
		address
		for path, _ in callgraph_files(build_directory)
		for location in callgraph_locations(path).values()
		if (
			address := known.by_declaration.get(
				Declaration(unit=path.name.removesuffix(".ci"), location=location)
			)
		)
		is not None
	)


@pytest.mark.image
@pytest.mark.parametrize(
	("fixture", "executable", "clones"),
	[("sensor-two-impl", "zephyr.elf", 7), ("counter-su", "zephyr.exe", 25)],
)
def test_every_clone_joins_exactly_one_ci_node_through_its_origins_declaration(
	zephyr_fixtures: Path, fixture: str, executable: str, clones: int
) -> None:
	program = load(zephyr_fixtures / fixture / "zephyr" / executable)
	joins = _joins(program, zephyr_fixtures / fixture)
	clone_addresses = frozenset(
		address
		for name, addresses in program.symbol_addresses.items()
		if re.search(r"\.(isra|constprop|part|cold)\b", name)
		for address in addresses
	)
	assert (
		len(clone_addresses),
		{joins[address] for address in clone_addresses},
		max(joins.values()),
	) == (
		clones,
		{1},
		1,
	)


@pytest.mark.image
def test_an_alias_joins_the_body_its_ci_records_under_another_name(zephyr_fixtures: Path) -> None:
	build_directory = zephyr_fixtures / "sensor-two-impl"
	program = load(build_directory / "zephyr" / "zephyr.elf")
	known = identities(program, stack_names(program))
	edges = {
		(frame_key(edge.caller), frame_key(edge.callee))
		for path, file_edges in callgraph_files(build_directory)
		for edge in canonical_edges(
			known, path.name.removesuffix(".ci"), callgraph_locations(path), file_edges
		)
	}
	assert {
		("z_impl_k_wakeup", "z_reschedule_locked"),
		("z_reschedule_locked", "z_reschedule"),
		("z_reschedule", "z_swap_irqlock"),
	} <= edges


@pytest.mark.image
def test_two_functions_sharing_a_name_become_two_nodes_each_with_its_own_frame(
	zephyr_fixtures: Path,
) -> None:
	build_directory = zephyr_fixtures / "sensor-two-impl"
	program = load(build_directory / "zephyr" / "zephyr.elf")
	known = identities(program, stack_names(program))
	assert (
		sorted(
			{
				name
				for path, file_edges in callgraph_files(build_directory)
				for edge in canonical_edges(
					known, path.name.removesuffix(".ci"), callgraph_locations(path), file_edges
				)
				for name in (edge.caller, edge.callee)
				if name.startswith("elapsed")
			}
		),
		sorted(
			(usage.function, usage.bytes)
			for path, usages in stack_usage_files(build_directory)
			for usage in canonical_frames(
				known, path.name.removesuffix(".su"), stack_usage_locations(path), usages
			)
			if usage.function.startswith("uart_stellaris_init")
		),
	) == (
		["elapsed@cortex_m_systick.c", "elapsed@timeout.c"],
		[("uart_stellaris_init@soc_config.c", 0), ("uart_stellaris_init@uart_stellaris.c", 20)],
	)


@pytest.mark.image
def test_names_that_collide_are_qualified_by_the_shortest_path_that_tells_them_apart(
	zephyr_fixtures: Path,
) -> None:
	assert sorted(
		name
		for name in stack_names(
			load(zephyr_fixtures / "counter-su" / "zephyr" / "zephyr.exe")
		).values()
		if name.startswith("main@")
	) == ["main@common/src/main.c", "main@counter/src/main.c"]


def test_a_colliding_function_without_a_declaration_is_qualified_by_its_address() -> None:
	assert sorted(
		stack_names(
			build_program(Machine.EM_X86_64, (("twin", 0x1000, 4), ("twin.isra.0", 0x2000, 4)))
		).values()
	) == ["twin@0x1000", "twin@0x2000"]
