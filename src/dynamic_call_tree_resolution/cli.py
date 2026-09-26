# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Command line interface for analyzing ELF images."""

from pathlib import Path  # noqa: TC003  # cyclopts evaluates Annotated[Path, ...] at runtime
from typing import Annotated

import msgspec
from cyclopts import App, Parameter

from dynamic_call_tree_resolution.callgraph import load_callgraph
from dynamic_call_tree_resolution.loader import load
from dynamic_call_tree_resolution.points_to import assignments
from dynamic_call_tree_resolution.report import build_report
from dynamic_call_tree_resolution.stack_analysis import worst_case_depths
from dynamic_call_tree_resolution.stack_usage import load_stack_usages

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
		print(msgspec.json.format(msgspec.json.encode(report).decode()))
		return
	for assignment in report.assignments:
		label = assignment.member_path or hex(assignment.slot_address)
		print(f"{label}: {', '.join(candidate.name for candidate in assignment.candidates)}")


@app.command
def stack(build_directory: Path) -> None:
	"""Print worst-case stack depths of a build directory (.su and .ci artifacts)."""
	reports = worst_case_depths(load_callgraph(build_directory), load_stack_usages(build_directory))
	for report in reports:
		flags = (" recursive" if report.recursive else "") + (
			" dynamic" if report.has_dynamic else ""
		)
		print(f"{report.entry}: {report.depth} bytes{flags}")


def main() -> None:
	"""Entry point installed as the ``dctr`` executable."""
	app()
