# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Joining ``.ci`` nodes and ``.su`` records to the image's functions (#155)."""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

import pytest

from dynamic_call_tree_resolution import Address, load
from dynamic_call_tree_resolution.callgraph import CallEdge, callgraph_files, callgraph_locations
from dynamic_call_tree_resolution.identity import Identities, canonical_edges, identities
from dynamic_call_tree_resolution.model import Declaration, SourceLocation
from dynamic_call_tree_resolution.stack_analysis import frame_key

if TYPE_CHECKING:
	from pathlib import Path

	from dynamic_call_tree_resolution.model import Program


def test_a_name_joins_its_function_by_declaration_then_by_symbol_and_otherwise_stays() -> None:
	known = Identities(
		by_declaration={
			Declaration(
				unit="a.c", location=SourceLocation(file="a.c", line=3, column=12)
			): Address(0x10),
		},
		by_symbol={"z_alias": Address(0x10), "collided": Address(0x30)},
		names={Address(0x10): "z_alias"},
	)
	assert canonical_edges(
		known,
		"a.c",
		{"a.c:alias": SourceLocation(file="a.c", line=3, column=12)},
		(
			CallEdge(caller="a.c:alias", callee="z_alias"),
			CallEdge(caller="caller", callee="collided"),
			CallEdge(caller="caller", callee="prebuilt"),
		),
	) == (
		CallEdge(caller="z_alias", callee="z_alias"),
		CallEdge(caller="caller", callee="collided"),
		CallEdge(caller="caller", callee="prebuilt"),
	)


@pytest.mark.image
def test_load_reads_each_functions_declaration_and_every_symbol_at_its_address(
	zephyr_fixtures: Path,
) -> None:
	program = load(zephyr_fixtures / "sensor-two-impl" / "zephyr" / "zephyr.elf")
	assert (
		[
			program.declarations[address]
			for address in sorted(program.symbol_addresses["bus_fault.isra.0"])
		],
		sorted(
			program.declarations[address].unit
			for address in program.symbol_addresses["uart_stellaris_init"]
		),
		program.symbol_addresses["z_reschedule"] == program.symbol_addresses["reschedule"],
		program.symbol_addresses["__l_vfprintf"].isdisjoint(program.declarations),
	) == (
		[Declaration(unit="fault.c", location=SourceLocation(file="fault.c", line=345, column=12))],
		["soc_config.c", "uart_stellaris.c"],
		True,
		True,
	)


def _joins(program: Program, build_directory: Path) -> Counter[Address]:
	known = identities(program)
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
	known = identities(load(build_directory / "zephyr" / "zephyr.elf"))
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
def test_two_functions_sharing_a_name_keep_their_ci_titles(zephyr_fixtures: Path) -> None:
	build_directory = zephyr_fixtures / "sensor-two-impl"
	known = identities(load(build_directory / "zephyr" / "zephyr.elf"))
	assert sorted(
		{
			title.rsplit("/", 1)[-1]
			for path, file_edges in callgraph_files(build_directory)
			for edge in canonical_edges(
				known, path.name.removesuffix(".ci"), callgraph_locations(path), file_edges
			)
			for title in (edge.caller, edge.callee)
			if frame_key(title) == "elapsed"
		}
	) == ["cortex_m_systick.c:elapsed", "timeout.c:elapsed"]
