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
from dynamic_call_tree_resolution.stack_analysis import expand_indirect_calls, worst_case_depths
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
def stack(build_directory: Path, *, elf: Path | None = None) -> None:
	"""Print worst-case stack depths of a build directory (.su and .ci artifacts).

	When an ELF is given, unresolved indirect call sites are expanded to
	every resolved function-pointer target.
	"""
	edges = load_callgraph(build_directory)
	indirect_sites = sum(edge.callee == "__indirect_call" for edge in edges)
	if elf is not None:
		program = load(elf)
		resolved = assignments(program)
		candidates = (
			program.functions[address].name
			for assignment in resolved
			for address in assignment.candidates
		)
		edges = expand_indirect_calls(edges, candidates)
		print(f"resolved slots: {len(resolved)} | indirect call sites: {indirect_sites}")
	reports = worst_case_depths(edges, load_stack_usages(build_directory))
	for report in reports:
		flags = (" recursive" if report.recursive else "") + (
			" dynamic" if report.has_dynamic else ""
		)
		print(f"{report.entry}: {report.depth} bytes{flags}")


def main() -> None:
	"""Entry point installed as the ``dctr`` executable."""
	app()
