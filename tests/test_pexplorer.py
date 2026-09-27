# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Tests for :mod:`dynamic_call_tree_resolution.pexplorer` and the comparison join."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import pytest

from dynamic_call_tree_resolution import (
	Address,
	ComparisonReport,
	PexplorerCallee,
	PexplorerFunction,
	PexplorerReport,
	Program,
	build_comparison,
	load,
	load_pexplorer,
)
from dynamic_call_tree_resolution.cli import compare
from dynamic_call_tree_resolution.pexplorer import DynamicSites, dynamic_sites_by_caller
from tests.programs import build_program

if TYPE_CHECKING:
	from pathlib import Path


def _write_report(path: Path, report: PexplorerReport) -> Path:
	path.write_bytes(msgspec.json.encode(report))
	return path


def test_load_pexplorer_parses_dynamic_callees(tmp_path: Path) -> None:
	report = PexplorerReport(
		functions=(
			PexplorerFunction(
				name="caller",
				address=0x1001,
				callees=(
					PexplorerCallee(call_from=0x1001, call_to=0x2000, dynamic=False),
					PexplorerCallee(call_from=0x1001, dynamic=True),
				),
			),
			PexplorerFunction(name="leaf", address=0x2000),
		)
	)
	loaded = load_pexplorer(_write_report(tmp_path / "report.json", report))
	assert loaded == report
	assert loaded.functions[1].callees == ()


def _main_pexplorer_report(main_address: int) -> PexplorerReport:
	return PexplorerReport(
		functions=(
			PexplorerFunction(
				name="main",
				address=main_address,
				callees=(
					PexplorerCallee(call_from=main_address, dynamic=True),
					PexplorerCallee(call_from=main_address, dynamic=True),
				),
			),
			PexplorerFunction(
				name="runtime_hook",
				address=0x9999,
				callees=(PexplorerCallee(call_from=0x9999, dynamic=True),),
			),
		)
	)


def test_comparison_joins_pexplorer_per_function(
	fixture_elfs: dict[str, Path], tmp_path: Path
) -> None:
	program = load(fixture_elfs["nopie"])
	main_address = next(
		function.address for function in program.functions.values() if function.name == "main"
	)
	report = load_pexplorer(
		_write_report(tmp_path / "report.json", _main_pexplorer_report(main_address))
	)
	comparison = build_comparison("nopie.elf", program, report)
	assert comparison.pexplorer_dynamic_sites == 3
	rows = {row.caller: row for row in comparison.function_comparisons}
	assert rows["main"].pexplorer_dynamic_sites == 2
	assert rows["main"].dctr_call_sites == 6
	assert rows["main"].dctr_resolved_sites == 5
	assert rows["main"].dctr_exact_sites == 5
	assert rows["runtime_hook"].pexplorer_dynamic_sites == 1
	assert rows["runtime_hook"].dctr_call_sites == 0
	assert rows["_start"].pexplorer_dynamic_sites == 0
	assert rows["_start"].dctr_call_sites == 1


def test_comparison_round_trip_with_pexplorer_fields(
	fixture_elfs: dict[str, Path], tmp_path: Path
) -> None:
	program = load(fixture_elfs["nopie"])
	main_address = next(
		function.address for function in program.functions.values() if function.name == "main"
	)
	report = load_pexplorer(
		_write_report(tmp_path / "report.json", _main_pexplorer_report(main_address))
	)
	comparison = build_comparison("nopie.elf", program, report)
	assert msgspec.json.decode(msgspec.json.encode(comparison), type=ComparisonReport) == comparison


def test_alias_callers_aggregate_instead_of_overwriting() -> None:
	report = PexplorerReport(
		functions=(
			PexplorerFunction(
				name="alias_a",
				address=0x1000,
				callees=(PexplorerCallee(call_from=0x1000, dynamic=True),),
			),
			PexplorerFunction(
				name="alias_b",
				address=0x1001,
				callees=(
					PexplorerCallee(call_from=0x1001, dynamic=True),
					PexplorerCallee(call_from=0x1001, dynamic=True),
				),
			),
			PexplorerFunction(
				name="quiet",
				address=0x2000,
				callees=(PexplorerCallee(call_from=0x2000, dynamic=False),),
			),
		)
	)
	assert dynamic_sites_by_caller(report) == {
		Address(0x1000): DynamicSites(names=("alias_a", "alias_b"), total=3)
	}


def test_missing_dynamic_flag_fails_loud(tmp_path: Path) -> None:
	report = tmp_path / "report.json"
	report.write_text(
		'{"functions": [{"name": "nodyn", "address": 4242, '
		'"callees": [{"from": 4242, "from_function_name": "nodyn"}]}]}'
	)
	with pytest.raises(msgspec.ValidationError, match="dynamic"):
		load_pexplorer(report)


def _arm_slot_program(caller_name: str) -> Program:
	body = bytes.fromhex("00 4b 98 47") + (0x2000).to_bytes(4, "little")
	return build_program(
		"EM_ARM",
		functions=((caller_name, 0x1001, 8), ("target", 0x2000, 4)),
		objects=(("slot", 0x3000, (0x2000).to_bytes(4, "little")),),
		sections={0x1000: body},
		pointer_size=4,
	)


def test_anonymous_rows_fall_back_to_the_pexplorer_name() -> None:
	program = _arm_slot_program("<anonymous>")
	report = PexplorerReport(
		functions=(
			PexplorerFunction(
				name="pex_name",
				address=0x1001,
				callees=(PexplorerCallee(call_from=0x1001, dynamic=True),),
			),
		)
	)
	comparison = build_comparison("anon.elf", program, report)
	rows = {row.caller: row for row in comparison.function_comparisons}
	assert "pex_name" in rows
	assert rows["pex_name"].address == 0x1000


def test_comparison_resolves_thumb_bit_only_callers() -> None:
	program = _arm_slot_program("thumb_caller")
	comparison = build_comparison("thumb.elf", program)
	rows = {row.caller: row for row in comparison.function_comparisons}
	assert rows["thumb_caller"].dctr_call_sites == 1


def test_cli_compare_joins_pexplorer_report(
	tmp_path: Path,
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	program = load(fixture_elfs["nopie"])
	main_address = next(
		function.address for function in program.functions.values() if function.name == "main"
	)
	report_path = _write_report(tmp_path / "report.json", _main_pexplorer_report(main_address))
	compare([fixture_elfs["nopie"]], pexplorer=report_path)
	output = capsys.readouterr().out
	assert "pexplorer" in output
	assert "runtime_hook" in output
	line = next(line for line in output.splitlines() if line.startswith("main "))
	assert line.endswith("2     6        5     5")
