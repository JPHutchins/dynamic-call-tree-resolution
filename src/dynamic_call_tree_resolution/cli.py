# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Command line interface for analyzing ELF images."""

import sys
from pathlib import Path  # noqa: TC003  # cyclopts evaluates Annotated[Path, ...] at runtime
from typing import Annotated, Final, assert_never

import msgspec
from cyclopts import App, Parameter
from elftools.common.exceptions import ELFError
from salix import Struct

from dynamic_call_tree_resolution.call_sites import extract_call_sites, per_caller_candidates
from dynamic_call_tree_resolution.callgraph import CallEdge, load_callgraph
from dynamic_call_tree_resolution.loader import defined_function_names, load
from dynamic_call_tree_resolution.pexplorer import load_pexplorer
from dynamic_call_tree_resolution.points_to import assignments, unresolved_slots
from dynamic_call_tree_resolution.report import (
	AnalysisSummary,
	StackEntryReport,
	build_comparison,
	build_report,
	slot_counts,
	stack_bound_report,
)
from dynamic_call_tree_resolution.stack_analysis import (
	INDIRECT_CALLEE,
	Bounded,
	StackReport,
	Unbounded,
	expand_indirect_calls,
	frame_key,
	worst_case_depths,
)
from dynamic_call_tree_resolution.stack_usage import load_stack_usages

app = App(name="dctr")

_MAX_ELF_SIZE: Final = 50 * 1024 * 1024

_NARROW_BY_SIGNATURE: Final = Parameter(
	help="narrow a site whose slot the image leaves unset to the functions of the "
	"slot's DWARF signature; unsound, since a cast defeats it"
)


def _oversized(path: Path) -> bool:
	return path.stat().st_size > _MAX_ELF_SIZE


def _warn_skipped(path: Path) -> None:
	print(
		f"warning: skipping {path} ({path.stat().st_size // 1024 // 1024} MiB; too large)",
		file=sys.stderr,
	)


def _render(report: StackReport) -> str:
	match report.bound:
		case Bounded(bytes=depth):
			return f"{report.entry}: {depth} bytes"
		case Unbounded() as bound:
			reasons = ", ".join(
				f"{reason}: {len(functions)}"
				for reason, functions in (
					("recursion", bound.recursion),
					("unmeasured", bound.unmeasured),
					("dynamic", bound.dynamic),
					("unresolved", bound.unresolved),
				)
				if functions
			)
			return f"{report.entry}: unbounded, at least {bound.at_least} bytes ({reasons})"
		case _ as unreachable:
			assert_never(unreachable)


def _keep(path: Path) -> bool:
	if not _oversized(path):
		return True
	_warn_skipped(path)
	return False


@app.command  # type: ignore[misc]
def analyze(
	elf: Annotated[Path, Parameter(help="ELF image to analyze")],
	*,
	json: Annotated[bool, Parameter(name=("--json", "-j"), help="emit JSON")] = False,
	narrow_by_signature: Annotated[bool, _NARROW_BY_SIGNATURE] = False,
) -> None:
	"""Print resolved function-pointer assignments and call sites of an ELF image."""
	program = load(elf)
	report = build_report(
		program,
		assignments(program),
		extract_call_sites(program),
		narrow_by_signature=narrow_by_signature,
	)
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


@app.command  # type: ignore[misc]
def compare(
	elfs: Annotated[
		list[Path], Parameter(help="ELF images or directories of ELF images to compare")
	],
	*,
	json: Annotated[bool, Parameter(name=("--json", "-j"), help="emit JSON")] = False,
	pexplorer: Annotated[
		Path | None,
		Parameter(
			help="pexplorer JSON report to join per function: every function with dynamic "
			"calls on either side gets a row comparing pexplorer's dynamic-call count with "
			"dctr's per-site candidate sets"
		),
	] = None,
	narrow_by_signature: Annotated[bool, _NARROW_BY_SIGNATURE] = False,
) -> None:
	"""Print a resolution rollup per ELF for cross-tool comparison."""
	paths: list[Path] = []
	for elf in elfs:
		if elf.is_dir():
			found = sorted(elf.glob("*.elf"))
			paths.extend(found)
			if not found:
				print(f"warning: {elf}: no .elf files", file=sys.stderr)
		else:
			paths.append(elf)
	paths = [path for path in paths if _keep(path)]
	pexplorer_report = load_pexplorer(pexplorer) if pexplorer is not None else None
	comparisons = [
		build_comparison(
			elf.name, load(elf), pexplorer_report, narrow_by_signature=narrow_by_signature
		)
		for elf in paths
	]
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


class _Expansion(Struct):
	expanded: tuple[CallEdge, ...]
	original: tuple[CallEdge, ...]
	indirect_sites: int
	resolved_slots: int
	image_functions: frozenset[str]


def _expand_from_elf(
	edges: tuple[CallEdge, ...], elf: Path, *, narrow_by_signature: bool
) -> _Expansion:
	indirect_sites = sum(edge.callee == INDIRECT_CALLEE for edge in edges)
	program = load(elf)
	resolved = assignments(program)
	targets_by_caller, fallback = per_caller_candidates(
		program, extract_call_sites(program), resolved, narrow_by_signature=narrow_by_signature
	)
	return _Expansion(
		expanded=expand_indirect_calls(edges, targets_by_caller, fallback),
		original=edges,
		indirect_sites=indirect_sites,
		resolved_slots=slot_counts(resolved, unresolved_slots(program, resolved)).resolved_slots,
		image_functions=frozenset(frame_key(name) for name in defined_function_names(elf)),
	)


def _in_image(reports: tuple[StackReport, ...], expansion: _Expansion) -> tuple[StackReport, ...]:
	return tuple(report for report in reports if report.entry in expansion.image_functions)


@app.command  # type: ignore[misc]
def stack(
	build_directory: Path,
	elf: Annotated[
		Path | None,
		Parameter(
			help="ELF image to expand indirect call sites against; entries the linker "
			"discarded are dropped and counted"
		),
	] = None,
	*,
	json: Annotated[bool, Parameter(name=("--json", "-j"), help="emit JSON")] = False,
	narrow_by_signature: Annotated[bool, _NARROW_BY_SIGNATURE] = False,
) -> None:
	"""Print worst-case stack depths of a build directory (.su and .ci artifacts)."""
	edges = load_callgraph(build_directory)
	expansion = (
		_expand_from_elf(edges, elf, narrow_by_signature=narrow_by_signature)
		if elf is not None
		else None
	)
	reports = worst_case_depths(
		expansion.expanded if expansion is not None else edges,
		load_stack_usages(build_directory),
		entry_edges=expansion.original if expansion is not None else None,
	)
	kept = _in_image(reports, expansion) if expansion is not None else reports
	if json:
		entries = tuple(
			StackEntryReport(entry=report.entry, bound=stack_bound_report(report.bound))
			for report in kept
		)
		print(msgspec.json.format(msgspec.json.encode(entries).decode()))
		return
	if expansion is not None:
		print(
			f"resolved slots: {expansion.resolved_slots} "
			f"| indirect call sites: {expansion.indirect_sites} "
			f"| not in the image: {len(reports) - len(kept)}"
			f"{' | narrowed by signature' if narrow_by_signature else ''}"
		)
	for report in kept:
		print(_render(report))


@app.command  # type: ignore[misc]
def summary(
	build_directory: Path,
	elf: Path,
	*,
	narrow_by_signature: Annotated[bool, _NARROW_BY_SIGNATURE] = False,
) -> None:
	"""Print a JSON summary combining resolution rates and worst-case stack depth."""
	program = load(elf)
	resolved = assignments(program)
	edges = load_callgraph(build_directory)
	expansion = _expand_from_elf(edges, elf, narrow_by_signature=narrow_by_signature)
	reports = worst_case_depths(
		expansion.expanded,
		load_stack_usages(build_directory),
		entry_edges=expansion.original,
	)
	kept = _in_image(reports, expansion)
	worst = next(iter(kept), StackReport(entry="", bound=Bounded(bytes=0)))
	unresolved = unresolved_slots(program, resolved)
	counts = slot_counts(resolved, unresolved)
	report = AnalysisSummary(
		resolved_slots=counts.resolved_slots,
		total_slots=counts.total_slots,
		unresolved_slots=len(unresolved),
		resolved_targets=counts.resolved_targets,
		indirect_call_sites=expansion.indirect_sites,
		total_functions=len(program.functions),
		entry_points=len(kept),
		discarded_entry_points=len(reports) - len(kept),
		worst_case_entry=worst.entry,
		worst_case=stack_bound_report(worst.bound),
	)
	print(msgspec.json.format(msgspec.json.encode(report).decode()))


def main() -> None:
	"""Entry point installed as the ``dctr`` executable.

	Raises:
		SystemExit: for usage errors and unparseable inputs.
	"""
	try:
		app()
	except (OSError, ELFError, ValueError, msgspec.ValidationError) as error:
		raise SystemExit(f"dctr: {error}") from None
