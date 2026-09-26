# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Tests for :mod:`dynamic_call_tree_resolution.cli`."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dynamic_call_tree_resolution.cli import analyze, main

EXPECTED_PATHS = {
	"ops_a.open",
	"ops_a.close",
	"ops_b.open",
	"ops_b.close",
	"dev_a.api.open",
	"dev_a.api.close",
	"dev_b.api.open",
	"dev_b.api.close",
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
	payload = json.loads(result.stdout)
	assert payload["resolved_slots"] == 11
	assert {assignment["member_path"] for assignment in payload["assignments"]} == EXPECTED_PATHS


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
	assert '"resolved_slots": 11' in capsys.readouterr().out
