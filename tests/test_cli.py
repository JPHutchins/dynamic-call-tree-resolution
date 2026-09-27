# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Tests for :mod:`dynamic_call_tree_resolution.cli`."""

import subprocess
import sys
from pathlib import Path

import msgspec
import pytest

from dynamic_call_tree_resolution import AnalysisReport, AnalysisSummary, ComparisonReport
from dynamic_call_tree_resolution.cli import analyze, compare, main, stack, summary
from tests.expected import EXPECTED_PATHS


def test_cli_analyze_json(fixture_elfs: dict[str, Path]) -> None:
	executable = Path(sys.executable).with_name("dctr")
	result = subprocess.run(
		[str(executable), "analyze", "--json", str(fixture_elfs["nopie"])],
		check=True,
		capture_output=True,
		text=True,
	)
	report = msgspec.json.decode(result.stdout, type=AnalysisReport)
	assert report.resolved_slots == 11
	assert {assignment.member_path for assignment in report.assignments} == EXPECTED_PATHS


def test_cli_analyze_plain_text(
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	analyze(fixture_elfs["nopie"])
	output = capsys.readouterr().out
	assert "dev_a.api.open: driver_a_open" in output
	assert "bss_cb: <unresolved>" in output
	assert "main@0x" in output


def test_cli_compare_json(fixture_elfs: dict[str, Path]) -> None:
	executable = Path(sys.executable).with_name("dctr")
	result = subprocess.run(
		[str(executable), "compare", "--json", str(fixture_elfs["nopie"])],
		check=True,
		capture_output=True,
		text=True,
	)
	comparisons = msgspec.json.decode(result.stdout, type=list[ComparisonReport])
	assert [comparison.elf for comparison in comparisons] == ["device_model.nopie.elf"]
	assert comparisons[0].call_sites == 7


def test_cli_compare_json_in_process(
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	compare([fixture_elfs["nopie"]], json=True)
	comparisons = msgspec.json.decode(capsys.readouterr().out, type=list[ComparisonReport])
	assert comparisons[0].call_sites == 7


def test_cli_compare_plain_text_and_directories(
	tmp_path: Path,
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	directory = tmp_path / "elfs"
	directory.mkdir()
	(directory / "a.elf").write_bytes(fixture_elfs["nopie"].read_bytes())
	(directory / "b.elf").write_bytes(fixture_elfs["nopie"].read_bytes())
	compare([directory])
	output = capsys.readouterr().out
	assert output.count("EM_X86_64") == 2
	assert "11/2/13" in output
	assert "5/5/7" in output


def test_cli_main_entry(
	fixture_elfs: dict[str, Path],
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	monkeypatch.setattr(sys, "argv", ["dctr", "analyze", "--json", str(fixture_elfs["nopie"])])
	with pytest.raises(SystemExit):
		main()
	report = msgspec.json.decode(capsys.readouterr().out, type=AnalysisReport)
	assert report.resolved_slots == 11


def test_cli_stack_plain_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
	build_directory = tmp_path / "build"
	nested = build_directory / "zephyr" / "CMakeFiles" / "zephyr.dir"
	nested.mkdir(parents=True)
	(nested / "main.c.ci").write_text(
		'graph: { edge: { sourcename: "main" targetname: "worker" } }\n'
	)
	(nested / "main.c.su").write_text("main.c:1:1:main\t16\tstatic\n")
	(nested / "worker.c.su").write_text("worker.c:2:1:worker\t32\tstatic\n")
	stack(build_directory)
	assert "main: 48 bytes" in capsys.readouterr().out


def test_cli_stack_with_elf_expands_indirect_sites(
	tmp_path: Path,
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	build_directory = tmp_path / "build"
	nested = build_directory / "zephyr" / "CMakeFiles" / "zephyr.dir"
	nested.mkdir(parents=True)
	(nested / "main.c.ci").write_text(
		'graph: { edge: { sourcename: "main" targetname: "__indirect_call" } }\n'
	)
	(nested / "main.c.su").write_text("main.c:1:1:main\t16\tstatic\n")
	(nested / "plain.c.su").write_text("plain.c:2:1:plain_target\t64\tstatic\n")
	stack(build_directory, elf=fixture_elfs["nopie"])
	output = capsys.readouterr().out
	assert "resolved slots: 11" in output
	assert "indirect call sites: 1" in output
	assert "main: 80 bytes" in output


def test_cli_stack_warns_when_indirect_edges_are_dropped(
	tmp_path: Path,
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	build_directory = tmp_path / "build"
	nested = build_directory / "zephyr" / "CMakeFiles" / "zephyr.dir"
	nested.mkdir(parents=True)
	(nested / "main.c.ci").write_text(
		'graph: { edge: { sourcename: "main" targetname: "__indirect_call" } }\n'
	)
	(nested / "main.c.su").write_text("main.c:1:1:main\t16\tstatic\n")
	stack(build_directory, elf=fixture_elfs["null"])
	output = capsys.readouterr()
	assert "warning: 1 of 1 indirect call edges have no candidates and were dropped" in output.err


def test_cli_bad_input_prints_one_line(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr(sys, "argv", ["dctr", "analyze", str(tmp_path / "missing.elf")])
	with pytest.raises(SystemExit) as error:
		main()
	assert str(error.value).startswith("dctr: ")


def test_cli_compare_warns_on_empty_directory(
	tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	empty = tmp_path / "empty"
	empty.mkdir()
	compare([empty])
	assert f"warning: {empty}: no .elf files" in capsys.readouterr().err


def test_cli_compare_skips_oversized_elfs(
	tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	directory = tmp_path / "elftest"
	directory.mkdir()
	oversized = directory / "oversized.elf"
	with oversized.open("wb") as stream:
		stream.truncate(51 * 1024 * 1024)
	compare([directory])
	assert "too large" in capsys.readouterr().err


def test_cli_stack_accepts_positional_elf(
	tmp_path: Path,
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	build_directory = tmp_path / "build"
	nested = build_directory / "zephyr" / "CMakeFiles" / "zephyr.dir"
	nested.mkdir(parents=True)
	(nested / "main.c.ci").write_text(
		'graph: { edge: { sourcename: "main" targetname: "__indirect_call" } }\n'
	)
	(nested / "main.c.su").write_text("main.c:1:1:main\t16\tstatic\n")
	(nested / "plain.c.su").write_text("plain.c:2:1:plain_target\t64\tstatic\n")
	stack(build_directory, fixture_elfs["nopie"])
	output = capsys.readouterr().out
	assert "resolved slots: 11" in output
	assert "main: 80 bytes" in output


def test_cli_summary_warns_when_indirect_edges_are_dropped(
	tmp_path: Path,
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	build_directory = tmp_path / "build"
	nested = build_directory / "zephyr" / "CMakeFiles" / "zephyr.dir"
	nested.mkdir(parents=True)
	(nested / "main.c.ci").write_text(
		'graph: { edge: { sourcename: "main" targetname: "__indirect_call" } }\n'
	)
	(nested / "main.c.su").write_text("main.c:1:1:main\t16\tstatic\n")
	summary(build_directory, fixture_elfs["null"])
	output = capsys.readouterr()
	assert "warning: 1 of 1 indirect call edges have no candidates and were dropped" in output.err
	report = msgspec.json.decode(output.out, type=AnalysisSummary)
	assert report.entry_points == 1


def test_cli_summary_json(
	tmp_path: Path,
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	build_directory = tmp_path / "build"
	nested = build_directory / "zephyr" / "CMakeFiles" / "zephyr.dir"
	nested.mkdir(parents=True)
	(nested / "main.c.ci").write_text(
		'graph: { edge: { sourcename: "main" targetname: "__indirect_call" } }\n'
	)
	(nested / "main.c.su").write_text("main.c:1:1:main\t16\tstatic\n")
	summary(build_directory, fixture_elfs["nopie"])
	report = msgspec.json.decode(capsys.readouterr().out, type=AnalysisSummary)
	assert report.resolved_slots == 11
	assert report.total_slots == 13
	assert report.unresolved_slots == 2
	assert report.indirect_call_sites == 1
	assert report.worst_case_entry == "main"
	assert report.worst_case_bytes == 16
