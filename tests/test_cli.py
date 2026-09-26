# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Tests for :mod:`dynamic_call_tree_resolution.cli`."""

import subprocess
import sys
from pathlib import Path

import msgspec
import pytest

from dynamic_call_tree_resolution import AnalysisReport
from dynamic_call_tree_resolution.cli import analyze, main, stack

EXPECTED_PATHS = {
	"ops_a.open",
	"ops_a.close",
	"ops_b.open",
	"ops_b.close",
	"dev_a.api.open",
	"dev_a.api.close",
	"dev_b.api.open",
	"dev_b.api.close",
	"dev_c.api.open",
	"dev_c.api.close",
	"dev_c.context.open",
	"dev_c.context.close",
	"dev_a.ops.init",
	"dev_b.ops.init",
	"dev_c.ops.init",
	"holder2.inner.fn",
	"holder.run",
	"node_a.fn",
	"plain_cb",
}


def test_cli_analyze_json(fixture_elfs: dict[str, Path]) -> None:
	executable = Path(sys.executable).with_name("dctr")
	result = subprocess.run(
		[str(executable), "analyze", "--json", str(fixture_elfs["nopie"])],
		check=True,
		capture_output=True,
		text=True,
	)
	report = msgspec.json.decode(result.stdout, type=AnalysisReport)
	assert report.resolved_slots == 19
	assert {assignment.member_path for assignment in report.assignments} == EXPECTED_PATHS


def test_cli_analyze_plain_text(
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	analyze(fixture_elfs["nopie"])
	assert "dev_a.api.open: driver_a_open" in capsys.readouterr().out


def test_cli_main_entry(
	fixture_elfs: dict[str, Path],
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	monkeypatch.setattr(sys, "argv", ["dctr", "analyze", "--json", str(fixture_elfs["nopie"])])
	with pytest.raises(SystemExit):
		main()
	report = msgspec.json.decode(capsys.readouterr().out, type=AnalysisReport)
	assert report.resolved_slots == 19


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
