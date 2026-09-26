# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Command line interface for analyzing ELF images."""

from pathlib import Path  # noqa: TC003  # cyclopts evaluates Annotated[Path, ...] at runtime
from typing import Annotated

from cyclopts import App, Parameter

from dynamic_call_tree_resolution.loader import load
from dynamic_call_tree_resolution.points_to import assignments
from dynamic_call_tree_resolution.report import build_report

app = App(name="dctr")


@app.command
def analyze(
	elf: Annotated[Path, Parameter(help="ELF image to analyze")],
	*,
	json: Annotated[bool, Parameter(name=("--json", "-j"), help="emit JSON")] = False,
) -> None:
	"""Print resolved function-pointer assignments of an ELF image."""
	program = load(elf)
	report = build_report(program, assignments(program))
	if json:
		print(report.model_dump_json(indent=2))
		return
	for assignment in report.assignments:
		label = assignment.member_path or hex(assignment.slot_address)
		print(f"{label}: {', '.join(candidate.name for candidate in assignment.candidates)}")


def main() -> None:
	"""Entry point installed as the ``dctr`` executable."""
	app()
