# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Command line interface for analyzing ELF images."""

from pathlib import Path  # noqa: TC003  # cyclopts evaluates Annotated[Path, ...] at runtime
from typing import Annotated

import msgspec
from cyclopts import App, Parameter

from dynamic_call_tree_resolution.call_sites import extract_call_sites, per_caller_candidates
from dynamic_call_tree_resolution.callgraph import load_callgraph
from dynamic_call_tree_resolution.loader import load
from dynamic_call_tree_resolution.pexplorer import load_pexplorer
from dynamic_call_tree_resolution.points_to import assignments, unresolved_slots
from dynamic_call_tree_resolution.report import AnalysisSummary, build_comparison, build_report
from dynamic_call_tree_resolution.stack_analysis import expand_indirect_calls, worst_case_depths
from dynamic_call_tree_resolution.stack_usage import load_stack_usages

app = App(name="dctr")


@app.command
def analyze(
	elf: Annotated[Path, Parameter(help="ELF image to analyze")],
	*,
	json: Annotated[bool, Parameter(name=("--json", "-j"), help="emit JSON")] = False,
) -> None:
	"""Print resolved function-pointer assignments and call sites of an ELF image."""
	program = load(elf)
	report = build_report(program, assignments(program), extract_call_sites(program))
	if json:
		print(msgspec.json.format(msgspec.json.encode(report).decode()))
		return
	for assignment in report.assignments:
		label = assignment.member_path or hex(assignment.slot_address)
		print(f"{label}: {', '.join(candidate.name for candidate in assignment.candidates)}")
	for slot in report.unresolved_slots:
		print(f"{slot.member_path}: <unresolved>")
	for site in report.call_sites:
		label = f"{site.caller}@{site.site_address:#x}"
		targets = ", ".join(candidate.name for candidate in site.candidates)
		print(f"{label}: {targets or '<unresolved>'}")


@app.command
def compare(
	elfs: Annotated[
		list[Path], Parameter(help="ELF images or directories of ELF images to compare")
	],
	*,
	json: Annotated[bool, Parameter(name=("--json", "-j"), help="emit JSON")] = False,
	pexplorer: Annotated[
		Path | None, Parameter(help="pexplorer JSON report to join per function")
	] = None,
) -> None:
	"""Print a resolution rollup per ELF for cross-tool comparison.

	With a pexplorer JSON report, every function with dynamic calls on
	either side gets a row comparing pexplorer's dynamic-call count with
	dctr's per-site candidate sets.
	"""
	paths = [
		path for elf in elfs for path in (sorted(elf.glob("*.elf")) if elf.is_dir() else (elf,))
	]
	pexplorer_report = load_pexplorer(pexplorer) if pexplorer is not None else None
	comparisons = [build_comparison(elf.name, load(elf), pexplorer_report) for elf in paths]
	if json:
		print(msgspec.json.format(msgspec.json.encode(comparisons).decode()))
		return
	print(f"{'elf':<46} {'machine':<10} {'functions':>9} {'slots r/u/t':>11} {'sites r/e/t':>12}")
	for comparison in comparisons:
		print(
			f"{comparison.elf:<46} {comparison.machine:<10} {comparison.functions:>9} "
			f"{f'{comparison.resolved_slots}/{comparison.unresolved_slots}/{comparison.total_slots}':>11} "
			f"{f'{comparison.resolved_call_sites}/{comparison.exact_call_sites}/{comparison.call_sites}':>12}"
		)
	if pexplorer_report is None:
		return
	print()
	print(f"{'function':<48} {'pexplorer':>9} {'dctr':>5} {'resolved':>8} {'exact':>5}")
	for comparison in comparisons:
		for row in comparison.function_comparisons:
			print(
				f"{row.caller:<48} {row.pexplorer_dynamic_sites:>9} "
				f"{row.dctr_call_sites:>5} {row.dctr_resolved_sites:>8} {row.dctr_exact_sites:>5}"
			)


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
		targets_by_caller, fallback = per_caller_candidates(
			program, extract_call_sites(program), resolved
		)
		edges = expand_indirect_calls(edges, targets_by_caller, fallback)
		print(f"resolved slots: {len(resolved)} | indirect call sites: {indirect_sites}")
	reports = worst_case_depths(edges, load_stack_usages(build_directory))
	for report in reports:
		flags = (" recursive" if report.recursive else "") + (
			" dynamic" if report.has_dynamic else ""
		)
		print(f"{report.entry}: {report.depth} bytes{flags}")


@app.command
def summary(build_directory: Path, elf: Path) -> None:
	"""Print a JSON summary combining resolution rates and worst-case stack depth."""
	program = load(elf)
	resolved = assignments(program)
	edges = load_callgraph(build_directory)
	indirect_sites = sum(edge.callee == "__indirect_call" for edge in edges)
	targets_by_caller, fallback = per_caller_candidates(
		program, extract_call_sites(program), resolved
	)
	expanded = expand_indirect_calls(edges, targets_by_caller, fallback)
	reports = worst_case_depths(expanded, load_stack_usages(build_directory))
	deepest = max(reports, key=lambda report: report.depth, default=None)
	unresolved = unresolved_slots(program, resolved)
	report = AnalysisSummary(
		resolved_slots=len(resolved),
		total_slots=len(
			{assignment.slot for assignment in resolved} | {slot.slot for slot in unresolved}
		),
		unresolved_slots=len(unresolved),
		resolved_targets=sum(len(assignment.candidates) for assignment in resolved),
		indirect_call_sites=indirect_sites,
		total_functions=len(program.functions),
		entry_points=len(reports),
		worst_case_bytes=deepest.depth if deepest is not None else 0,  # pragma: no branch
		worst_case_entry=deepest.entry if deepest is not None else "",  # pragma: no branch
	)
	print(msgspec.json.format(msgspec.json.encode(report).decode()))


def main() -> None:
	"""Entry point installed as the ``dctr`` executable."""
	app()
