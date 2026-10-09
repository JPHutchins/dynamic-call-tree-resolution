# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Narrowing a call through a struct field to what the field holds (#159)."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import msgspec
import pytest
from salix import replace

from dynamic_call_tree_resolution import Machine, load, resolve
from dynamic_call_tree_resolution.call_sites import call_site_candidates
from dynamic_call_tree_resolution.cli import analyze, stack
from dynamic_call_tree_resolution.descriptors import (
	Descriptors,
	Field,
	Function,
	Null,
	Parameter,
	Site,
	Store,
	Variable,
	load_descriptors,
)
from dynamic_call_tree_resolution.field_narrowing import (
	FieldNarrowing,
	field_narrowings,
	field_targets,
	narrowed,
	narrowing_at,
)
from dynamic_call_tree_resolution.loader import line_spans
from dynamic_call_tree_resolution.model import (
	Address,
	EmbeddedStructMember,
	FunctionPointerMember,
	LineSpan,
	SourceLocation,
	StructureLayout,
)
from dynamic_call_tree_resolution.report import (
	BoundedStack,
	StackEntryReport,
	UnboundedStack,
	resolved_by_slot,
)
from dynamic_call_tree_resolution.vsa import address_taken
from tests.programs import build_program
from tests.toolchains import FIXTURES, build_cortex_m3, run_cortex_m3

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path

	from dynamic_call_tree_resolution.model import Program


def _run_member(offset: int) -> FunctionPointerMember:
	return FunctionPointerMember(kind="function_pointer", name="run", offset=offset, signature=None)


def _program() -> Program:
	program = build_program(
		Machine.EM_X86_64,
		(
			("caller", 0x1000, 0x10),
			("initial", 0x3000, 1),
			("embedded", 0x3010, 1),
			("stored", 0x3020, 1),
		),
		objects=(
			("table", 0x5000, (0x3000).to_bytes(8, "little")),
			("holder", 0x5100, bytes(8) + (0x3010).to_bytes(8, "little")),
			("wild", 0x5200, (0x5000).to_bytes(8, "little")),
		),
	)
	return replace(
		program,
		objects={
			address: replace(data_object, type_name=type_name)
			for address, data_object in program.objects.items()
			for type_name in (
				{"table": "struct ops", "holder": "struct holder", "wild": "struct wild"}[
					data_object.name
				],
			)
		},
		layouts={
			"struct ops": StructureLayout(members=(_run_member(0),), size=8),
			"struct holder": StructureLayout(
				members=(
					EmbeddedStructMember(
						kind="embedded_struct",
						name="ops",
						offset=8,
						type_name="struct ops",
						members=(_run_member(0),),
					),
				),
				size=16,
			),
			"struct wild": StructureLayout(members=(_run_member(0),), size=8),
		},
		symbol_addresses={
			"caller": frozenset({Address(0x1000)}),
			"stored": frozenset({Address(0x3020)}),
		},
	)


def _store(record: str, member: str, value: Function | Null | Parameter) -> Store:
	return Store(
		unit="a.c",
		function="install",
		location=SourceLocation(file="a.c", line=1, column=1),
		place=Field(record=record, member=member),
		value=value,
	)


def test_a_field_holds_its_initializers_and_its_stored_functions_unless_a_store_is_unknown() -> (
	None
):
	program = _program()
	assert {
		f"{field.record}.{field.member}": sorted(
			program.functions[target].name for target in targets
		)
		for field, targets in field_targets(
			program,
			(
				_store("struct ops", "run", Function(symbol="stored")),
				_store("struct ops", "run", Null()),
				_store("struct stored_only", "run", Function(symbol="stored")),
				_store("struct parameter", "run", Function(symbol="stored")),
				_store("struct parameter", "run", Parameter(name="callback")),
				_store("struct missing", "run", Function(symbol="inlined_away")),
				Store(
					unit="a.c",
					function="install",
					location=SourceLocation(file="a.c", line=2, column=1),
					place=Variable(symbol="hook"),
					value=Parameter(name="callback"),
				),
			),
		).items()
	} == {
		"struct ops.run": ["embedded", "initial", "stored"],
		"struct stored_only.run": ["stored"],
	}


def test_a_span_narrows_only_where_its_functions_plugin_sites_agree_on_a_narrowable_field() -> None:
	program = _program()
	location = SourceLocation(file="a.c", line=7, column=3)
	elsewhere = SourceLocation(file="a.c", line=9, column=3)
	spans = field_narrowings(
		program,
		Descriptors(
			sites=(
				Site(
					unit="a.c",
					function="caller",
					location=location,
					callee=Field(record="struct ops", member="run"),
				),
				Site(
					unit="a.c",
					function="caller",
					location=elsewhere,
					callee=Field(record="struct ops", member="run"),
				),
				Site(
					unit="a.c",
					function="caller",
					location=elsewhere,
					callee=Parameter(name="callback"),
				),
			),
			stores=(),
		),
		(
			LineSpan(start=Address(0x1004), end=Address(0x1008), location=location),
			LineSpan(start=Address(0x1008), end=Address(0x100C), location=elsewhere),
			LineSpan(start=Address(0x2000), end=Address(0x2004), location=location),
		),
	)
	ops = FieldNarrowing(
		field=Field(record="struct ops", member="run"),
		targets=frozenset({Address(0x3000), Address(0x3010)}),
	)
	assert (
		[narrowing_at(spans, Address(address)) for address in (0x1003, 0x1004, 0x1007, 0x1008)],
		narrowed(frozenset(), spans, Address(0x1006)),
		narrowed(frozenset({Address(0x3020)}), spans, Address(0x1006)),
		narrowed(frozenset(), (), Address(0x1006)),
	) == (
		[None, ops, ops, None],
		ops.targets,
		frozenset({Address(0x3020)}),
		frozenset(),
	)


@pytest.mark.image
def test_analyze_narrows_the_device_api_calls_to_what_their_fields_hold(
	zephyr_fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	analyze(zephyr_fixtures / "sensor-two-impl" / "zephyr" / "zephyr.elf", narrow_by_field=True)
	assert [
		line for line in capsys.readouterr().out.splitlines() if "narrowed by field" in line
	] == [
		(
			"bmi160_init@0x17b2: bmi160_bus_ready_i2c "
			"(narrowed by field struct bmi160_bus_io.ready, unsound under casts)"
		),
		(
			"emul_init_for_bus@0x3154: adt7420_emul_init, emul_bosch_bmi160_init "
			"(narrowed by field struct emul.init, unsound under casts)"
		),
		(
			"i2c_emul_transfer@0x31da: i2c_emul_transfer "
			"(narrowed by field struct i2c_driver_api.transfer, unsound under casts)"
		),
		(
			"i2c_emul_transfer@0x31fe: adt7420_emul_transfer_i2c, bmi160_emul_transfer_i2c "
			"(narrowed by field struct i2c_emul_api.transfer, unsound under casts)"
		),
		(
			"i2c_emul_transfer@0x3212: adt7420_emul_transfer_i2c, bmi160_emul_transfer_i2c "
			"(narrowed by field struct i2c_emul_api.transfer, unsound under casts)"
		),
		(
			"i2c_write@0x3258: i2c_emul_transfer "
			"(narrowed by field struct i2c_driver_api.transfer, unsound under casts)"
		),
		(
			"i2c_write_read.constprop.0@0x32ac: i2c_emul_transfer "
			"(narrowed by field struct i2c_driver_api.transfer, unsound under casts)"
		),
		(
			"bmi160_read_i2c@0x376a: i2c_emul_transfer "
			"(narrowed by field struct i2c_driver_api.transfer, unsound under casts)"
		),
		(
			"bmi160_write_i2c@0x37aa: i2c_emul_transfer "
			"(narrowed by field struct i2c_driver_api.transfer, unsound under casts)"
		),
		(
			"bmi160_read@0x37bc: bmi160_read_i2c "
			"(narrowed by field struct bmi160_bus_io.read, unsound under casts)"
		),
		(
			"bmi160_write@0x3830: bmi160_write_i2c "
			"(narrowed by field struct bmi160_bus_io.write, unsound under casts)"
		),
	]


def _recursing(measured: tuple[str, ...]) -> UnboundedStack:
	return UnboundedStack(
		at_least_bytes=16 + 184 + 36,
		recursion=("i2c_emul_transfer",),
		unmeasured=(),
		dynamic=(),
		unresolved=(),
		measured=measured,
		stack_reservation_bytes=16,
		exception_frame_bytes=36,
		narrowed_by_field=("i2c_emul_transfer",),
	)


@pytest.mark.image
@pytest.mark.parametrize(
	("fixture", "bounds"),
	[
		(
			"sensor-threads",
			{
				"motion_tid": BoundedStack(
					bytes=16 + 184 + 36,
					measured=("__aeabi_ldivmod", "__aeabi_read_tp", "memset"),
					stack_reservation_bytes=16,
					exception_frame_bytes=36,
					narrowed_by_field=("spi_emul_io",),
				),
				"thermal_tid": _recursing(("__aeabi_ldivmod", "__aeabi_read_tp", "memset")),
			},
		),
		(
			"sensor-two-impl",
			{
				"motion_tid": _recursing(("__aeabi_ldivmod", "__aeabi_read_tp")),
				"thermal_tid": _recursing(("__aeabi_ldivmod", "__aeabi_read_tp")),
			},
		),
	],
)
def test_each_threads_tree_stays_below_its_own_bus_and_recurses_only_through_i2c_forwarding(
	zephyr_fixtures: Path,
	capsys: pytest.CaptureFixture[str],
	fixture: str,
	bounds: Mapping[str, BoundedStack | UnboundedStack],
) -> None:
	stack(
		zephyr_fixtures / fixture,
		elf=zephyr_fixtures / fixture / "zephyr" / "zephyr.elf",
		narrow_by_field=True,
		json=True,
	)
	assert {
		row.entry: row.bound
		for row in msgspec.json.decode(capsys.readouterr().out, type=tuple[StackEntryReport, ...])
		if row.entry in bounds
	} == bounds


@pytest.mark.parametrize("level", ["-O2", "-Os"])
def test_a_function_stored_through_a_cast_is_missed_by_the_narrowing_and_kept_by_the_default(
	tmp_path: Path, level: str
) -> None:
	elf = build_cortex_m3(
		(FIXTURES / "reproducers" / "field_cast.c",), tmp_path / "image.elf", "-g", level
	)
	descriptors = load_descriptors(FIXTURES / "reproducers" / "field_cast.descriptors.txt")
	program = load(elf)
	resolution = resolve(program)
	(site,) = (
		site for site in resolution.sites if program.functions[site.caller_address].name == "run"
	)
	names = {address: function.name for address, function in program.functions.items()}
	assert (
		run_cortex_m3(elf, "x", "y").stdout.split(),
		call_site_candidates(program, site, resolved_by_slot(resolution.assignments)),
		{
			names[target]
			for target in narrowed(
				frozenset(),
				field_narrowings(
					program,
					descriptors,
					line_spans(elf, frozenset(site.location for site in descriptors.sites)),
				),
				site.site_address,
			)
		},
		"target" in {names[address] for address in address_taken(program)},
	) == (["@run", "target"], frozenset(), {"decoy"}, True)


def test_analyze_reads_the_descriptors_beside_the_elf(
	tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	elf = build_cortex_m3(
		(FIXTURES / "reproducers" / "field_cast.c",), tmp_path / "image.elf", "-g", "-O2"
	)
	shutil.copy(
		FIXTURES / "reproducers" / "field_cast.descriptors.txt", tmp_path / "descriptors.txt"
	)
	analyze(elf, narrow_by_field=True)
	assert [line for line in capsys.readouterr().out.splitlines() if line.startswith("run@")] == [
		f"run@{_run_site(elf):#x}: decoy (narrowed by field struct ops.run, unsound under casts)"
	]


def _run_site(elf: Path) -> int:
	program = load(elf)
	(site,) = (
		site
		for site in resolve(program).sites
		if program.functions[site.caller_address].name == "run"
	)
	return site.site_address
