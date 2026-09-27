# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Command line interface for analyzing ELF images."""

import sys
from collections.abc import Mapping  # noqa: TC003  # evaluated at runtime in _dropped_indirect_edges' signature
from pathlib import Path  # noqa: TC003  # cyclopts evaluates Annotated[Path, ...] at runtime
from typing import Annotated

import msgspec
from cyclopts import App, Parameter
from elftools.common.exceptions import ELFError
from salix import Struct

from dynamic_call_tree_resolution.call_sites import extract_call_sites, per_caller_candidates
from dynamic_call_tree_resolution.callgraph import CallEdge, load_callgraph
from dynamic_call_tree_resolution.loader import load
from dynamic_call_tree_resolution.pexplorer import load_pexplorer
from dynamic_call_tree_resolution.points_to import assignments, unresolved_slots
from dynamic_call_tree_resolution.report import (
	AnalysisSummary,
	build_comparison,
	build_report,
	slot_counts,
)
from dynamic_call_tree_resolution.stack_analysis import (
	INDIRECT_CALLEE,
	StackReport,
	expand_indirect_calls,
	frame_key,
	worst_case_depths,
)
from dynamic_call_tree_resolution.stack_usage import load_stack_usages

app = App(name="dctr")

_MAX_ELF_SIZE = 50 * 1024 * 1024


def _oversized(path: Path) -> bool:
	return path.stat().st_size > _MAX_ELF_SIZE


def _warn_skipped(path: Path) -> None:
	print(
		f"warning: skipping {path} ({path.stat().st_size // 1024 // 1024} MiB; too large)",
		file=sys.stderr,
	)


def _depth(report: StackReport) -> int:
	return report.depth


def _keep(path: Path) -> bool:
	"""Whether to analyze ``path``, warning on stderr when skipping it."""
	if not _oversized(path):
		return True
	_warn_skipped(path)
	return False


def _dropped_indirect_edges(
	edges: tuple[CallEdge, ...],
	targets_by_caller: Mapping[str, frozenset[str]],
	fallback: frozenset[str],
) -> int:
	"""Indirect edges left with no candidates after expansion."""
	return sum(
		edge.callee == INDIRECT_CALLEE
		and not (targets_by_caller.get(frame_key(edge.caller), frozenset()) | fallback)
		for edge in edges
	)


@app.command  # type: ignore[misc]
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


@app.command  # type: ignore[misc]
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


class _Expansion(Struct):
	"""One ELF-expanded call-graph: the edges, the originals, and the resolution."""

	expanded: tuple[CallEdge, ...]
	original: tuple[CallEdge, ...]
	targets_by_caller: Mapping[str, frozenset[str]]
	fallback: frozenset[str]
	indirect_sites: int
	resolved_slots: int


def _expand_from_elf(edges: tuple[CallEdge, ...], elf: Path) -> _Expansion:
	"""Resolve and expand the indirect edges of ``edges`` against an ELF image."""
	indirect_sites = sum(edge.callee == INDIRECT_CALLEE for edge in edges)
	program = load(elf)
	resolved = assignments(program)
	targets_by_caller, fallback = per_caller_candidates(
		program, extract_call_sites(program), resolved
	)
	return _Expansion(
		expanded=expand_indirect_calls(edges, targets_by_caller, fallback),
		original=edges,
		targets_by_caller=targets_by_caller,
		fallback=fallback,
		indirect_sites=indirect_sites,
		resolved_slots=slot_counts(resolved, unresolved_slots(program, resolved)).resolved_slots,
	)


def _warn_dropped_indirect_edges(expansion: _Expansion) -> None:
	"""Warn when any indirect edge has no candidates and was dropped."""
	if dropped := _dropped_indirect_edges(
		expansion.original, expansion.targets_by_caller, expansion.fallback
	):
		print(
			f"warning: {dropped} of {expansion.indirect_sites} indirect call edges "
			"have no candidates and were dropped",
			file=sys.stderr,
		)


@app.command  # type: ignore[misc]
def stack(build_directory: Path, elf: Path | None = None) -> None:
	"""Print worst-case stack depths of a build directory (.su and .ci artifacts).

	When an ELF is given, unresolved indirect call sites are expanded to
	every resolved function-pointer target.
	"""
	edges = load_callgraph(build_directory)
	expansion = _expand_from_elf(edges, elf) if elf is not None else None
	if expansion is not None:
		print(
			f"resolved slots: {expansion.resolved_slots} "
			f"| indirect call sites: {expansion.indirect_sites}"
		)
		_warn_dropped_indirect_edges(expansion)
	reports = worst_case_depths(
		expansion.expanded if expansion is not None else edges,
		load_stack_usages(build_directory),
	)
	for report in reports:
		flags = (
			(" recursive" if report.recursive else "")
			+ (" dynamic" if report.has_dynamic else "")
			+ (f" unmeasured: {report.unmeasured}" if report.unmeasured else "")
		)
		print(f"{report.entry}: {report.depth} bytes{flags}")


@app.command  # type: ignore[misc]
def summary(build_directory: Path, elf: Path) -> None:
	"""Print a JSON summary combining resolution rates and worst-case stack depth."""
	program = load(elf)
	resolved = assignments(program)
	edges = load_callgraph(build_directory)
	expansion = _expand_from_elf(edges, elf)
	_warn_dropped_indirect_edges(expansion)
	reports = worst_case_depths(expansion.expanded, load_stack_usages(build_directory))
	deepest = max(reports, key=_depth, default=None)
	unresolved = unresolved_slots(program, resolved)
	counts = slot_counts(resolved, unresolved)
	report = AnalysisSummary(
		resolved_slots=counts.resolved_slots,
		total_slots=counts.total_slots,
		unresolved_slots=len(unresolved),
		resolved_targets=counts.resolved_targets,
		indirect_call_sites=expansion.indirect_sites,
		total_functions=len(program.functions),
		entry_points=len(reports),
		worst_case_bytes=deepest.depth if deepest is not None else 0,  # pragma: no branch
		worst_case_entry=deepest.entry if deepest is not None else "",  # pragma: no branch
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
