# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Command line interface for analyzing ELF images."""

import sys
from pathlib import Path  # noqa: TC003  # cyclopts evaluates Annotated[Path, ...] at runtime
from typing import TYPE_CHECKING, Annotated, Final, assert_never

import msgspec
from cyclopts import App, Parameter
from elftools.common.exceptions import ELFError
from salix import Struct

from dynamic_call_tree_resolution.call_sites import own_targets, per_caller_candidates, resolve
from dynamic_call_tree_resolution.callgraph import CallEdge, load_callgraph
from dynamic_call_tree_resolution.loader import defined_function_names, elf_machine, load
from dynamic_call_tree_resolution.model import Machine, Residue
from dynamic_call_tree_resolution.pexplorer import PexplorerReport, load_pexplorer
from dynamic_call_tree_resolution.points_to import unresolved_slots
from dynamic_call_tree_resolution.report import (
	AnalysisSummary,
	ComparisonReport,
	PathStepReport,
	ReferrerReport,
	RtosReport,
	SlotCounts,
	StackEntryReport,
	StackPathReport,
	build_comparison,
	build_report,
	referrers_report,
	rtos_report,
	slot_counts,
	stack_bound_report,
)
from dynamic_call_tree_resolution.rtos import RtosChoice, rtos_model
from dynamic_call_tree_resolution.stack_analysis import (
	INDIRECT_CALLEE,
	Bounded,
	PathStep,
	StackGraph,
	StackReport,
	Unbounded,
	deepest_path,
	expand_indirect_calls,
	frame_key,
	own_thread_edges,
	stack_graph,
	stack_reports,
	thread_edges,
	thread_frames,
	thread_reports,
)
from dynamic_call_tree_resolution.stack_usage import StackUsage, load_stack_usages

if TYPE_CHECKING:
	from collections.abc import Iterator, Mapping

	from dynamic_call_tree_resolution.call_sites import ProgramResolution
	from dynamic_call_tree_resolution.model import Program, RtosModel, UnresolvedSlot

app = App(name="dctr")

_MAX_ELF_SIZE: Final = 50 * 1024 * 1024

_NARROW_BY_SIGNATURE: Final = Parameter(
	help="narrow a site whose slot the image leaves unset to the functions of the "
	"slot's DWARF signature; unsound, since a cast defeats it"
)

_RTOS: Final = Parameter(
	help="the RTOS whose threads start their entries with known arguments: auto detects "
	"it, and none models no RTOS"
)


def _oversized(path: Path) -> bool:
	return path.stat().st_size > _MAX_ELF_SIZE


def _skip_reason(path: Path) -> str | None:
	if _oversized(path):
		return f"{path.stat().st_size // 1024 // 1024} MiB; too large"
	if (machine := elf_machine(path)) not in Machine:
		return f"{machine}; unsupported machine"
	return None


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


def _residue_label(residue: Residue) -> str:
	match residue:
		case Residue.ROM_NULL:
			return "<null>"
		case Residue.ROM_NON_FUNCTION:
			return "<not a function>"
		case Residue.RAM_NULL | Residue.RAM_UNINITIALIZED | Residue.RAM_INITIALIZED:
			return "<unresolved>"
		case _ as unreachable:
			assert_never(unreachable)


def _render_step(step: PathStep) -> str:
	return (
		f"{step.function} +{step.frame} = {step.cumulative} bytes"
		f"{' via ' + ', '.join(sorted(step.edge)) if step.edge else ''}"
		f"{' (' + ', '.join(sorted(step.flags)) + ')' if step.flags else ''}"
	)


def _step_report(step: PathStep) -> PathStepReport:
	return PathStepReport(
		function=step.function,
		frame_bytes=step.frame,
		cumulative_bytes=step.cumulative,
		edge=tuple(sorted(step.edge)),
		flags=tuple(sorted(step.flags)),
	)


def _rtos_lines(report: RtosReport) -> tuple[str, ...]:
	return (
		(
			f"rtos: {report.name} (detected: {', '.join(report.evidence)})",
			*(
				f"thread {thread.name}: {thread.entry.name} "
				f"({'seeded from its record' if thread.seeded else 'not seeded: its address is taken elsewhere'})"
				for thread in report.threads
			),
		)
		if report.evidence
		else ()
	)


def _keep(path: Path) -> bool:
	match _skip_reason(path):
		case None:
			return True
		case str() as reason:
			print(f"warning: skipping {path} ({reason})", file=sys.stderr)
			return False
		case _ as unreachable:
			assert_never(unreachable)


@app.command  # type: ignore[misc]
def analyze(
	elf: Annotated[Path, Parameter(help="ELF image to analyze")],
	*,
	json: Annotated[bool, Parameter(name=("--json", "-j"), help="emit JSON")] = False,
	narrow_by_signature: Annotated[bool, _NARROW_BY_SIGNATURE] = False,
	rtos: Annotated[RtosChoice, _RTOS] = RtosChoice.AUTO,
) -> None:
	"""Print resolved function-pointer assignments and call sites of an ELF image."""
	program = load(elf)
	model = rtos_model(program, rtos)
	resolution = resolve(program, model)
	report = build_report(
		program,
		resolution.assignments,
		resolution.sites,
		narrow_by_signature=narrow_by_signature,
		rtos=rtos_report(program, model, resolution.seeded),
	)
	if json:
		print(msgspec.json.format(msgspec.json.encode(report).decode()))
		return
	for line in _rtos_lines(report.rtos):
		print(line)
	for assignment in report.assignments:
		label = assignment.member_path or hex(assignment.slot_address)
		print(f"{label}: {', '.join(candidate.name for candidate in assignment.candidates)}")
	for slot in report.unresolved_slots:
		print(f"{slot.member_path}: {_residue_label(slot.residue)}")
	for item in report.not_enumerated:
		print(f"{item.member_path}: <not enumerated: {item.reason}>")
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
	rtos: Annotated[RtosChoice, _RTOS] = RtosChoice.AUTO,
) -> None:
	"""Print a resolution rollup per ELF for cross-tool comparison."""
	paths = [path for path in tuple(_elf_paths(elfs)) if _keep(path)]
	pexplorer_report = load_pexplorer(pexplorer) if pexplorer is not None else None
	comparisons = [
		_comparison(elf, pexplorer_report, narrow_by_signature=narrow_by_signature, rtos=rtos)
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


def _elf_paths(elfs: list[Path]) -> Iterator[Path]:
	for elf in elfs:
		if elf.is_dir():
			found = sorted(elf.glob("*.elf"))
			if not found:
				print(f"warning: {elf}: no .elf files", file=sys.stderr)
			yield from found
		else:
			yield elf


def _comparison(
	elf: Path, pexplorer: PexplorerReport | None, *, narrow_by_signature: bool, rtos: RtosChoice
) -> ComparisonReport:
	program = load(elf)
	return build_comparison(
		elf.name,
		program,
		pexplorer,
		narrow_by_signature=narrow_by_signature,
		rtos=rtos_model(program, rtos),
	)


class _Expansion(Struct):
	expanded: tuple[CallEdge, ...]
	in_image_edges: tuple[CallEdge, ...]
	frames: tuple[StackUsage, ...]
	threads: frozenset[str]
	own_edges: Mapping[str, tuple[CallEdge, ...]]
	"""Each thread's tree from its own analysis, by thread."""
	indirect_sites: int
	program: Program
	rtos: RtosModel
	counts: SlotCounts
	unresolved: tuple[UnresolvedSlot, ...]
	image_functions: frozenset[str]


def _expand_from_elf(
	edges: tuple[CallEdge, ...],
	frames: tuple[StackUsage, ...],
	elf: Path,
	*,
	narrow_by_signature: bool,
	rtos: RtosChoice,
) -> _Expansion:
	indirect_sites = sum(edge.callee == INDIRECT_CALLEE for edge in edges)
	program = load(elf)
	image_functions = frozenset(frame_key(name) for name in defined_function_names(elf))
	in_image_edges = tuple(edge for edge in edges if frame_key(edge.caller) in image_functions)
	model = rtos_model(program, rtos)
	resolution = resolve(program, model)
	targets_by_caller, fallback = per_caller_candidates(
		program, resolution.sites, resolution.assignments, narrow_by_signature=narrow_by_signature
	)
	unresolved = unresolved_slots(program, resolution.assignments)
	threads_edges, threads_frames = _thread_graph(program, model, in_image_edges, frames)
	expanded = expand_indirect_calls(edges, targets_by_caller, fallback)
	return _Expansion(
		expanded=(*expanded, *threads_edges),
		in_image_edges=(*in_image_edges, *threads_edges),
		frames=(*frames, *threads_frames),
		threads=frozenset(thread.name for thread in model.threads),
		own_edges={
			thread: (*expanded, *own)
			for thread, own in _own_thread_graphs(
				program, model, resolution, in_image_edges, targets_by_caller, fallback
			).items()
		},
		indirect_sites=indirect_sites,
		program=program,
		rtos=model,
		counts=slot_counts(resolution.assignments, unresolved),
		unresolved=unresolved,
		image_functions=image_functions,
	)


def _thread_graph(
	program: Program,
	model: RtosModel,
	edges: tuple[CallEdge, ...],
	frames: tuple[StackUsage, ...],
) -> tuple[tuple[CallEdge, ...], tuple[StackUsage, ...]]:
	entries_by_thread = {
		thread.name: frame_key(program.functions[thread.entry].name) for thread in model.threads
	}
	match model.trampoline:
		case None:
			return (), ()
		case str() as trampoline:
			return (
				thread_edges(edges, trampoline, entries_by_thread),
				thread_frames(frames, trampoline, entries_by_thread),
			)
		case _ as unreachable:
			assert_never(unreachable)


def _own_thread_graphs(
	program: Program,
	model: RtosModel,
	resolution: ProgramResolution,
	edges: tuple[CallEdge, ...],
	targets_by_caller: Mapping[str, frozenset[str]],
	fallback: frozenset[str],
) -> Mapping[str, tuple[CallEdge, ...]]:
	entries = {thread.name: thread.entry for thread in model.threads}
	match model.trampoline:
		case None:
			return {}
		case str() as trampoline:
			return {
				thread: own_thread_edges(
					edges,
					trampoline,
					thread,
					frame_key(program.functions[entries[thread]].name),
					own_targets(program, sites, resolution.assignments),
					targets_by_caller,
					fallback,
				)
				for thread, sites in resolution.threads.items()
			}
		case _ as unreachable:
			assert_never(unreachable)


def _in_image(reports: tuple[StackReport, ...], expansion: _Expansion) -> tuple[StackReport, ...]:
	return tuple(
		report
		for report in reports
		if report.entry in expansion.image_functions or report.entry in expansion.threads
	)


@app.command  # type: ignore[misc]
def stack(
	build_directory: Path,
	elf: Annotated[
		Path | None,
		Parameter(
			help="ELF image to expand indirect call sites against; calls from functions the "
			"linker discarded are ignored, and discarded entries are dropped and counted"
		),
	] = None,
	*,
	json: Annotated[bool, Parameter(name=("--json", "-j"), help="emit JSON")] = False,
	narrow_by_signature: Annotated[bool, _NARROW_BY_SIGNATURE] = False,
	rtos: Annotated[RtosChoice, _RTOS] = RtosChoice.AUTO,
	path: Annotated[
		str | None,
		Parameter(
			help="print only this entry, with its deepest path: each function's frame and "
			"cumulative bytes, the edge it is called through, and what makes it unbounded"
		),
	] = None,
) -> None:
	"""Print worst-case stack depths of a build directory (.su and .ci artifacts)."""
	edges = load_callgraph(build_directory)
	frames = load_stack_usages(build_directory)
	expansion = (
		_expand_from_elf(edges, frames, elf, narrow_by_signature=narrow_by_signature, rtos=rtos)
		if elf is not None
		else None
	)
	graph = stack_graph(
		expansion.expanded if expansion is not None else edges,
		expansion.frames if expansion is not None else frames,
		entry_edges=expansion.in_image_edges if expansion is not None else None,
	)
	reports = stack_reports(graph)
	own_graphs = _own_graphs(expansion)
	kept = (
		_in_image(thread_reports(reports, own_graphs), expansion)
		if expansion is not None
		else reports
	)
	shown = kept if path is None else _entry_report(kept, path)
	steps = None if path is None else deepest_path(own_graphs.get(path, graph), path)
	if json:
		print(msgspec.json.format(msgspec.json.encode(_stack_document(shown, steps)).decode()))
		return
	if expansion is not None:
		print(
			f"resolved slots: {expansion.counts.resolved_slots} "
			f"| indirect call sites: {expansion.indirect_sites} "
			f"| not in the image: {len(reports) - len(kept)}"
			f"{' | narrowed by signature' if narrow_by_signature else ''}"
			f"{f' | rtos: {expansion.rtos.name}' if expansion.rtos.evidence else ''}"
		)
	for report in shown:
		print(_render(report))
	for step in steps or ():
		print(_render_step(step))


def _own_graphs(expansion: _Expansion | None) -> Mapping[str, StackGraph]:
	return (
		{
			thread: stack_graph(edges, expansion.frames)
			for thread, edges in expansion.own_edges.items()
		}
		if expansion is not None
		else {}
	)


def _stack_document(
	shown: tuple[StackReport, ...], steps: tuple[PathStep, ...] | None
) -> tuple[StackEntryReport, ...] | StackPathReport:
	return (
		tuple(
			StackEntryReport(entry=report.entry, bound=stack_bound_report(report.bound))
			for report in shown
		)
		if steps is None
		else StackPathReport(
			entry=shown[0].entry,
			bound=stack_bound_report(shown[0].bound),
			path=tuple(_step_report(step) for step in steps),
		)
	)


def _entry_report(reports: tuple[StackReport, ...], entry: str) -> tuple[StackReport, ...]:
	report = next((report for report in reports if report.entry == entry), None)
	if report is None:
		raise ValueError(f"{entry} is not a stack entry")
	return (report,)


@app.command  # type: ignore[misc]
def summary(
	build_directory: Path,
	elf: Path,
	*,
	narrow_by_signature: Annotated[bool, _NARROW_BY_SIGNATURE] = False,
	rtos: Annotated[RtosChoice, _RTOS] = RtosChoice.AUTO,
) -> None:
	"""Print a JSON summary combining resolution rates and worst-case stack depth."""
	expansion = _expand_from_elf(
		load_callgraph(build_directory),
		load_stack_usages(build_directory),
		elf,
		narrow_by_signature=narrow_by_signature,
		rtos=rtos,
	)
	reports = stack_reports(
		stack_graph(expansion.expanded, expansion.frames, entry_edges=expansion.in_image_edges)
	)
	kept = _in_image(thread_reports(reports, _own_graphs(expansion)), expansion)
	worst = next(iter(kept), StackReport(entry="", bound=Bounded(bytes=0)))
	report = AnalysisSummary(
		resolved_slots=expansion.counts.resolved_slots,
		total_slots=expansion.counts.total_slots,
		unresolved_slots=len(expansion.unresolved),
		resolved_targets=expansion.counts.resolved_targets,
		indirect_call_sites=expansion.indirect_sites,
		total_functions=len(expansion.program.functions),
		entry_points=len(kept),
		discarded_entry_points=len(reports) - len(kept),
		worst_case_entry=worst.entry,
		worst_case=stack_bound_report(worst.bound),
		rtos=expansion.rtos.name,
	)
	print(msgspec.json.format(msgspec.json.encode(report).decode()))


@app.command  # type: ignore[misc]
def referrers(
	elf: Annotated[Path, Parameter(help="ELF image to analyze")],
	*,
	json: Annotated[bool, Parameter(name=("--json", "-j"), help="emit JSON")] = False,
) -> None:
	"""Print every address-taken function and each place that holds or computes its address."""
	report = referrers_report(load(elf))
	if json:
		print(msgspec.json.format(msgspec.json.encode(report).decode()))
		return
	for function in report:
		print(f"{function.name}: {', '.join(map(_render_referrer, function.referrers))}")


def _render_referrer(referrer: ReferrerReport) -> str:
	match referrer.holder:
		case None:
			return f"{referrer.slot:#x}"
		case str() as holder:
			return f"{holder}@{referrer.slot:#x}"
		case _ as unreachable:
			assert_never(unreachable)


def main() -> None:
	"""Entry point installed as the ``dctr`` executable.

	Raises:
		SystemExit: for usage errors and unparseable inputs.
	"""
	try:
		app()
	except (OSError, ELFError, ValueError, msgspec.ValidationError) as error:
		raise SystemExit(f"dctr: {error}") from None
