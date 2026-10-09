# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Command line interface for analyzing ELF images."""

import sys
from collections import Counter
from itertools import accumulate
from pathlib import Path  # noqa: TC003  # cyclopts evaluates Annotated[Path, ...] at runtime
from typing import TYPE_CHECKING, Annotated, Final, assert_never

import msgspec
from cyclopts import App, Parameter
from elftools.common.exceptions import ELFError
from salix import Struct, replace

from dynamic_call_tree_resolution.call_sites import (
	calls_nothing,
	narrowed_targets,
	own_targets,
	per_caller_candidates,
	resolve,
)
from dynamic_call_tree_resolution.callgraph import (
	CallEdge,
	EdgeKind,
	callgraph_files,
	callgraph_locations,
)
from dynamic_call_tree_resolution.descriptors import load_descriptors
from dynamic_call_tree_resolution.field_narrowing import field_narrowings
from dynamic_call_tree_resolution.identity import (
	Identities,
	canonical_edges,
	canonical_frames,
	identities,
	stack_name,
	stack_names,
)
from dynamic_call_tree_resolution.linker import (
	Linkage,
	Membership,
	held_artifacts,
	linker_records,
)
from dynamic_call_tree_resolution.loader import (
	defined_function_names,
	elf_machine,
	line_spans,
	load,
)
from dynamic_call_tree_resolution.model import Address, Dead, Machine, Residue, Unreached
from dynamic_call_tree_resolution.pexplorer import PexplorerReport, load_pexplorer
from dynamic_call_tree_resolution.points_to import unresolved_slots
from dynamic_call_tree_resolution.report import (
	AnalysisSummary,
	CallSiteReport,
	ComparisonReport,
	NestingReport,
	PathStepReport,
	ReferrerReport,
	RtosReport,
	SignatureReport,
	SlotCounts,
	StackEntryReport,
	StackPathReport,
	build_comparison,
	build_report,
	nesting_report,
	referrers_report,
	rtos_report,
	slot_counts,
	stack_bound_report,
)
from dynamic_call_tree_resolution.rtos import RtosChoice, rtos_model
from dynamic_call_tree_resolution.rtos.zephyr import NO_KCONFIG, Kconfig, kconfig, priority_levels
from dynamic_call_tree_resolution.stack_analysis import (
	INDIRECT_CALLEE,
	AssumedFrame,
	Bounded,
	Frame,
	Handled,
	LevelSource,
	MeasuredFrame,
	Nesting,
	OwnFrame,
	PathStep,
	PriorityLevels,
	StackGraph,
	StackReport,
	Unbounded,
	deepest_path,
	entry_report,
	expand_indirect_calls,
	frame_key,
	interrupt_stack_report,
	own_thread_edges,
	stack_graph,
	stack_reports,
	thread_edges,
	thread_frames,
	thread_reports,
)
from dynamic_call_tree_resolution.stack_usage import (
	StackUsage,
	stack_usage_files,
	stack_usage_locations,
)
from dynamic_call_tree_resolution.vsa.abi import normalized
from dynamic_call_tree_resolution.vsa.frames import (
	CodeMeasure,
	OwnCode,
	code_depths,
	code_measure,
	own_frames,
)
from dynamic_call_tree_resolution.vsa.lattice import Known, Top
from dynamic_call_tree_resolution.vsa.links import linked_calls
from dynamic_call_tree_resolution.vsa.vectors import (
	HARD_FAULT,
	NMI,
	RESET,
	exception_handlers,
	hardware_handlers,
)

if TYPE_CHECKING:
	from collections.abc import Iterator, Mapping

	from dynamic_call_tree_resolution.call_sites import ProgramResolution
	from dynamic_call_tree_resolution.field_narrowing import NarrowedSpan
	from dynamic_call_tree_resolution.model import CallSite, Program, RtosModel, UnresolvedSlot

app = App(name="dctr")

_MAX_ELF_SIZE: Final = 50 * 1024 * 1024

_NARROW_BY_SIGNATURE: Final = Parameter(
	help="narrow a site whose slot the image leaves unset to the functions of the "
	"slot's DWARF signature; unsound, since a cast defeats it"
)

_NARROW_BY_FIELD: Final = Parameter(
	help="narrow a site that loads its callee from a struct field to what that field holds: "
	"its initializers in the image and the functions stored into it, from the descriptors.txt "
	"beside the ELF; unsound, since a cast or a memcpy can store a function unseen"
)

_ASSUME_NO_RECURSION: Final = Parameter(
	help="assume this function never calls itself, and drop its call to itself; repeat it "
	"for each function. Only a direct self-call is dropped, and the rows that rest on it "
	"say so"
)

_ASSUME_UNWRITTEN: Final = Parameter(
	help="assume the program never writes this data object after it loads, so a load from "
	"it reads its value in the image even where a store's address is unknown; repeat it "
	"for each object. Nothing checks it, and the rows that rest on it say so"
)

_ASSUME_FRAME: Final = Parameter(
	help="assume FUNCTION=BYTES is the frame of a function nothing measures; repeat it for "
	"each function. Nothing checks it, and the rows that rest on it say so"
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
		case Bounded(bytes=depth) as bound:
			notes = _notes(
				(
					("measured", bound.measured),
					("assumed no recursion", bound.assumed_no_recursion),
					("narrowed by field", bound.narrowed_by_field),
					("narrowed by signature", bound.narrowed_by_signature),
					("assumed unwritten", bound.assumed_unwritten),
					("assumed frame", bound.assumed_frames),
				),
				_thread_stack_additions(bound),
				(*_nesting_notes(report.nesting), *_stack_notes(report)),
			)
			return f"{report.entry}: {depth} bytes" + (f" ({notes})" if notes else "")
		case Unbounded() as bound:
			notes = _notes(
				(
					("recursion", bound.recursion),
					("unmeasured", bound.unmeasured),
					("measured", bound.measured),
					("assumed no recursion", bound.assumed_no_recursion),
					("narrowed by field", bound.narrowed_by_field),
					("narrowed by signature", bound.narrowed_by_signature),
					("assumed unwritten", bound.assumed_unwritten),
					("assumed frame", bound.assumed_frames),
					("dynamic", bound.dynamic),
					("unresolved", bound.unresolved),
				),
				_thread_stack_additions(bound),
				(*_nesting_notes(report.nesting), *_stack_notes(report)),
			)
			return f"{report.entry}: unbounded, at least {bound.at_least} bytes ({notes})"
		case _ as unreachable:
			assert_never(unreachable)


def _notes(
	counts: tuple[tuple[str, frozenset[str]], ...],
	additions: tuple[tuple[str, int], ...],
	nesting: tuple[str, ...],
) -> str:
	return ", ".join(
		(
			*(f"{name}: {len(functions)}" for name, functions in counts if functions),
			*(f"{name}: {added} bytes" for name, added in additions if added),
			*nesting,
		)
	)


def _thread_stack_additions(bound: Bounded | Unbounded) -> tuple[tuple[str, int], ...]:
	return (
		("stack reservation", bound.stack_reservation),
		("exception frame", bound.exception_frame),
	)


def _thread_stack_steps(
	shown: tuple[StackReport, ...], steps: tuple[PathStep, ...] | None
) -> tuple[str, ...]:
	return tuple(
		f"({name}) +{added} = {steps[-1].cumulative + total} bytes"
		for report in shown[:1]
		if steps
		for additions in (_thread_stack_additions(report.bound),)
		for (name, added), total in zip(
			additions, accumulate(added for _, added in additions), strict=True
		)
		if added
	)


def _path_steps(report: StackReport, graph: StackGraph) -> tuple[PathStep, ...]:
	return deepest_path(graph, report.entry) if report.nesting is None else ()


def _nesting_steps(
	shown: tuple[StackReport, ...], steps: tuple[PathStep, ...] | None
) -> tuple[str, ...]:
	return tuple(
		line
		for report in shown[:1]
		if steps is not None and report.nesting is not None
		for nesting in (report.nesting,)
		for line in (
			(
				f"(exception {nesting.base.exception}) {nesting.base.handler} "
				f"+{nesting.base.depth} = {nesting.base.depth} bytes"
			),
			*(
				f"(exception {nested.exception}) {nested.handler} "
				f"+{nesting.exception_frame} +{nested.depth} "
				f"= {total} bytes"
				for nested, total in zip(
					nesting.chain,
					tuple(
						accumulate(
							(nesting.exception_frame + nested.depth for nested in nesting.chain),
							initial=nesting.base.depth,
						)
					)[1:],
					strict=True,
				)
			),
		)
	)


def _nesting_document(nesting: Nesting | None) -> NestingReport | None:
	return nesting_report(nesting) if nesting is not None else None


def _nesting_notes(nesting: Nesting | None) -> tuple[str, ...]:
	return (
		(
			f"nested exceptions: {len(nesting.chain)}",
			f"priority levels: {nesting.levels.count} ({nesting.levels.source})",
			f"exception frame: {nesting.exception_frame} bytes each",
		)
		if nesting is not None
		else ()
	)


def _stack_notes(report: StackReport) -> tuple[str, ...]:
	return (
		*((f"stack: {report.stack_size} bytes",) if report.stack_size is not None else ()),
		*(f"margin: {margin} bytes" for margin in (_margin(report),) if margin is not None),
	)


def _margin(report: StackReport) -> int | None:
	match report.bound, report.stack_size:
		case Bounded(bytes=depth), int() as stack_size:
			return stack_size - depth
		case ((Bounded() | Unbounded()), (int() | None)):
			return None
		case _ as unreachable:
			assert_never(unreachable)


def _signature_text(signature: SignatureReport) -> str:
	return f"{signature.return_type} ({', '.join(signature.parameters)})"


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
	narrow_by_field: Annotated[bool, _NARROW_BY_FIELD] = False,
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
		narrowed_by_field=_narrowed_by_field(elf, program) if narrow_by_field else (),
		rtos=rtos_report(program, model, resolution.seeded),
		fallback=resolution.fallback,
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
		narrowing = (
			f" (narrowed by field {site.field}, unsound under casts)"
			if site.field is not None
			else f" (narrowed by signature {_signature_text(site.signature)}, unsound under casts)"
			if site.signature is not None
			else " (dispatch, from the RTOS model)"
			if site.dispatch
			else ""
		)
		print(f"{label}: {targets or _no_candidates(site)}{narrowing}")


def _no_candidates(site: CallSiteReport) -> str:
	return (
		f"<external: {site.external}>"
		if site.external is not None
		else "<null>"
		if site.null
		else "<dead>"
		if site.dead
		else "<unresolved>"
	)


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
	narrow_by_field: Annotated[bool, _NARROW_BY_FIELD] = False,
	rtos: Annotated[RtosChoice, _RTOS] = RtosChoice.AUTO,
) -> None:
	"""Print a resolution rollup per ELF for cross-tool comparison."""
	paths = [path for path in tuple(_elf_paths(elfs)) if _keep(path)]
	pexplorer_report = load_pexplorer(pexplorer) if pexplorer is not None else None
	comparisons = [
		_comparison(
			elf,
			pexplorer_report,
			narrow_by_signature=narrow_by_signature,
			narrow_by_field=narrow_by_field,
			rtos=rtos,
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
	elf: Path,
	pexplorer: PexplorerReport | None,
	*,
	narrow_by_signature: bool,
	narrow_by_field: bool,
	rtos: RtosChoice,
) -> ComparisonReport:
	program = load(elf)
	return build_comparison(
		elf.name,
		program,
		pexplorer,
		narrow_by_signature=narrow_by_signature,
		narrowed_by_field=_narrowed_by_field(elf, program) if narrow_by_field else (),
		rtos=rtos_model(program, rtos),
	)


def _narrowed_by_field(elf: Path, program: Program) -> tuple[NarrowedSpan, ...]:
	descriptors = load_descriptors(elf.parent / "descriptors.txt")
	return field_narrowings(
		program,
		descriptors,
		line_spans(elf, frozenset(site.location for site in descriptors.sites)),
	)


class Expansion(Struct):
	"""A build's call graph and frames, its indirect calls expanded against its ELF image."""

	expanded: tuple[CallEdge, ...]
	in_image_edges: tuple[CallEdge, ...]
	frames: tuple[Frame, ...]
	threads: frozenset[str]
	hardware_roots: frozenset[str]
	"""The handlers only the vector table holds, by their names in the stack graph."""
	exceptions: Mapping[int, str]
	"""Each exception's handler on an interrupt stack the RTOS model names, by exception
	number."""
	priority_levels: PriorityLevels
	own_edges: Mapping[str, tuple[CallEdge, ...]]
	"""Each thread's tree from its own analysis, by thread."""
	indirect_sites: int
	program: Program
	rtos: RtosModel
	resolution: ProgramResolution
	counts: SlotCounts
	unresolved: tuple[UnresolvedSlot, ...]
	image_functions: frozenset[str]
	membership: Membership
	dropped: Mapping[Linkage, frozenset[str]]
	"""The entries the linker left out of the image, by why; empty under name membership."""
	phantom_libcalls: frozenset[str]
	"""Libcalls a kept function's ``.ci`` records but the final link did not keep."""
	assumed_unwritten: AssumedUnwritten


class AssumedUnwritten(Struct):
	"""What ``--assume-unwritten`` changed."""

	objects: tuple[str, ...]
	contradicted: tuple[str, ...]
	"""The objects a tracked store writes after all."""
	callers: frozenset[str]
	"""The callers whose indirect calls it changed, by frame key."""
	thread_callers: Mapping[str, frozenset[str]]
	"""The same, in each thread's own analysis."""


class _Artifacts(Struct):
	"""The ``.ci`` edges and ``.su`` frames a stack graph is built from."""

	edges: tuple[CallEdge, ...]
	frames: tuple[StackUsage, ...]
	membership: Membership
	held: frozenset[str]
	"""The frame keys of the functions the artifacts hold; the linker's choice under its
	membership."""
	dropped: Mapping[Linkage, frozenset[str]]
	phantom_libcalls: frozenset[str]
	"""Libcalls a kept function's ``.ci`` records but the final link did not keep."""


class _Image(Struct):
	"""An ELF image, its analysis model and its functions' names in the stack graph."""

	elf: Path
	program: Program
	names: Mapping[Address, str]


def _image(elf: Path | None) -> _Image | None:
	return _loaded(elf) if elf is not None else None


def _loaded(elf: Path) -> _Image:
	program = load(elf)
	return _Image(elf=elf, program=program, names=stack_names(program))


def _artifacts(build_directory: Path, image: _Image | None) -> _Artifacts:
	known = identities(image.program, image.names) if image is not None else None
	match linker_records(image.elf) if image is not None else None:
		case None:
			return _named_artifacts(
				known,
				callgraph_files(build_directory),
				stack_usage_files(build_directory),
				Membership.NAMES,
				{},
				frozenset(),
			)
		case records:
			held = held_artifacts(
				records,
				build_directory,
				callgraph_files(build_directory),
				stack_usage_files(build_directory),
			)
			return _named_artifacts(
				known,
				held.callgraphs,
				held.usages,
				Membership.LINKER,
				held.dropped,
				held.phantom_libcalls,
			)


def _named_artifacts(
	known: Identities | None,
	callgraphs: tuple[tuple[Path, tuple[CallEdge, ...]], ...],
	usages: tuple[tuple[Path, tuple[StackUsage, ...]], ...],
	membership: Membership,
	dropped: Mapping[Linkage, frozenset[str]],
	phantom_libcalls: frozenset[str],
) -> _Artifacts:
	edges = tuple(
		edge for path, file_edges in callgraphs for edge in _named_edges(known, path, file_edges)
	)
	frames = tuple(
		frame for path, file_frames in usages for frame in _named_frames(known, path, file_frames)
	)
	return _Artifacts(
		edges=edges,
		frames=frames,
		membership=membership,
		held=frozenset(
			(
				*(frame_key(edge.caller) for edge in edges),
				*(frame_key(frame.function) for frame in frames),
			)
		),
		dropped=dropped,
		phantom_libcalls=phantom_libcalls,
	)


def _named_edges(
	known: Identities | None, path: Path, edges: tuple[CallEdge, ...]
) -> tuple[CallEdge, ...]:
	return (
		canonical_edges(known, path.name.removesuffix(".ci"), callgraph_locations(path), edges)
		if known is not None
		else edges
	)


def _named_frames(
	known: Identities | None, path: Path, frames: tuple[StackUsage, ...]
) -> tuple[StackUsage, ...]:
	return (
		canonical_frames(known, path.name.removesuffix(".su"), stack_usage_locations(path), frames)
		if known is not None
		else frames
	)


def elf_expansion(
	build_directory: Path,
	elf: Path,
	*,
	narrow_by_signature: bool,
	narrow_by_field: bool,
	rtos: RtosChoice,
	assume_unwritten: tuple[str, ...] = (),
) -> Expansion:
	"""Expand a build directory's indirect calls against its ELF image."""
	image = _loaded(elf)
	return _expand_from_elf(
		_artifacts(build_directory, image),
		image,
		narrow_by_signature=narrow_by_signature,
		narrow_by_field=narrow_by_field,
		rtos=rtos,
		assume_unwritten=assume_unwritten,
	)


def _expand_from_elf(
	artifacts: _Artifacts,
	image: _Image,
	*,
	narrow_by_signature: bool,
	narrow_by_field: bool,
	rtos: RtosChoice,
	assume_unwritten: tuple[str, ...] = (),
) -> Expansion:
	image_functions = _image_functions(artifacts, image)
	in_image_edges = tuple(
		edge for edge in artifacts.edges if frame_key(edge.caller) in image_functions
	)
	indirect_sites = sum(edge.callee == INDIRECT_CALLEE for edge in in_image_edges)
	model = rtos_model(image.program, rtos, _kconfig(image.elf))
	spans = _unwritten_spans(image.program, assume_unwritten)
	resolution = resolve(image.program, model, assumed_unwritten=spans)
	narrowed_by_field = _narrowed_by_field(image.elf, image.program) if narrow_by_field else ()
	hardware_roots = frozenset(
		stack_name(image.program, image.names, handler)
		for handler in hardware_handlers(image.program)
	)
	exceptions = (
		{
			number: frame_key(stack_name(image.program, image.names, handler))
			for number, handler in exception_handlers(image.program).items()
		}
		if model.interrupt_stack is not None
		else {}
	)
	targets_by_caller, fallback = per_caller_candidates(
		image.program,
		resolution.sites,
		resolution.assignments,
		narrow_by_signature=narrow_by_signature,
		narrowed_by_field=narrowed_by_field,
		names=image.names,
		fallback=resolution.fallback,
	)
	narrowed_by_caller = narrowed_targets(
		image.program,
		resolution.sites,
		resolution.assignments,
		narrow_by_signature=narrow_by_signature,
		narrowed_by_field=narrowed_by_field,
		names=image.names,
		fallback=resolution.fallback,
	)
	unresolved = unresolved_slots(image.program, resolution.assignments)
	measure = code_measure(image.program)
	register = _register_callers(image, measure, resolution.sites, artifacts.frames)
	unrecorded = _unrecorded_sites(image, resolution.sites, (*in_image_edges, *register.edges))
	expanded = expand_indirect_calls(
		(*artifacts.edges, *register.edges, *unrecorded),
		targets_by_caller,
		fallback,
		narrowed_by_caller=narrowed_by_caller,
	)
	binary = _binary_edges(image, measure, expanded, artifacts.frames)
	threads_edges, threads_frames = _thread_graph(
		image,
		model,
		(*in_image_edges, *binary, *register.edges),
		(*artifacts.frames, *register.frames),
	)
	code = _code_measures(
		image,
		measure,
		(*expanded, *binary, *threads_edges),
		(*artifacts.frames, *threads_frames, *register.frames),
		hardware_roots | frozenset(exceptions.values()),
		frozenset(),
	)
	return Expansion(
		expanded=(*expanded, *binary, *code.edges, *threads_edges),
		in_image_edges=(*in_image_edges, *unrecorded, *threads_edges),
		frames=(*artifacts.frames, *threads_frames, *register.frames, *code.frames),
		threads=frozenset(thread.name for thread in model.threads)
		| frozenset(thread.name for thread in model.system_threads),
		hardware_roots=hardware_roots,
		exceptions=exceptions,
		priority_levels=_priority_levels(image.elf, exceptions),
		own_edges={
			thread: (*expanded, *binary, *code.edges, *own)
			for thread, own in _own_thread_graphs(
				image,
				model,
				resolution,
				(*in_image_edges, *unrecorded, *binary),
				_Targets(
					by_caller=targets_by_caller,
					narrowed_by_caller=narrowed_by_caller,
					fallback=fallback,
					narrowed_by_field=narrowed_by_field,
					narrow_by_signature=narrow_by_signature,
				),
			).items()
		},
		indirect_sites=indirect_sites,
		program=image.program,
		rtos=model,
		resolution=resolution,
		counts=slot_counts(resolution.assignments, unresolved),
		unresolved=unresolved,
		image_functions=image_functions,
		membership=artifacts.membership,
		dropped=artifacts.dropped,
		phantom_libcalls=artifacts.phantom_libcalls,
		assumed_unwritten=_assumed_unwritten(image, model, resolution, assume_unwritten),
	)


def _unwritten_spans(program: Program, names: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
	return tuple(span for name in names for span in (_unwritten_span(program, name),))


def _unwritten_span(program: Program, name: str) -> tuple[int, int]:
	match tuple(obj for obj in program.objects.values() if obj.name == name):
		case (found,):
			return found.address, found.address + found.size
		case ():
			raise ValueError(f"--assume-unwritten: no data object is named {name}")
		case several:
			raise ValueError(
				f"--assume-unwritten: {len(several)} data objects are named {name}, at "
				f"{', '.join(hex(obj.address) for obj in several)}"
			)


def _assumed_unwritten(
	image: _Image, model: RtosModel, resolution: ProgramResolution, names: tuple[str, ...]
) -> AssumedUnwritten:
	if not names:
		return AssumedUnwritten(objects=(), contradicted=(), callers=frozenset(), thread_callers={})
	baseline = resolve(image.program, model)
	return AssumedUnwritten(
		objects=names,
		contradicted=tuple(
			name
			for name in names
			for start, end in (_unwritten_span(image.program, name),)
			if any(start - image.program.pointer_size < key < end for key in resolution.written)
		),
		callers=_changed_callers(image, baseline.sites, resolution.sites),
		thread_callers={
			thread: _changed_callers(image, baseline.threads[thread].sites, own.sites)
			for thread, own in resolution.threads.items()
		},
	)


def _changed_callers(
	image: _Image, before: tuple[CallSite, ...], after: tuple[CallSite, ...]
) -> frozenset[str]:
	targets = {(site.caller_address, site.site_address): _target_key(site) for site in before}
	return frozenset(
		frame_key(stack_name(image.program, image.names, site.caller_address))
		for site in after
		if targets.get((site.caller_address, site.site_address)) != _target_key(site)
	)


def _target_key(site: CallSite) -> tuple[str, frozenset[Address]]:
	match site.target:
		case Known(values=values):
			return "known", values
		case Top() | Unreached() | Dead() as other:
			return type(other).__name__, frozenset()
		case _ as unreachable:
			assert_never(unreachable)


def _call_pair(edge: CallEdge) -> tuple[str, str]:
	return edge.caller, edge.callee


def _binary_edges(
	image: _Image, measure: CodeMeasure, edges: tuple[CallEdge, ...], frames: tuple[StackUsage, ...]
) -> tuple[CallEdge, ...]:
	framed = frozenset(frame_key(frame.function) for frame in frames)
	known = frozenset((frame_key(edge.caller), frame_key(edge.callee)) for edge in edges)
	calls = tuple(
		(caller, callee)
		for call in linked_calls(image.program)
		if call.caller is not None
		for caller in (normalized(call.caller, image.program.machine),)
		for callee in (normalized(call.callee, image.program.machine),)
		if caller in image.names and callee in image.names
	)
	measured = code_depths(
		measure,
		frozenset(caller for caller, _ in calls if frame_key(image.names[caller]) not in framed),
	).keys()
	return tuple(
		sorted(
			{
				CallEdge(
					caller=image.names[caller], callee=image.names[callee], kind=EdgeKind.BINARY
				)
				for caller, callee in calls
				if caller not in measured
				and (frame_key(image.names[caller]), frame_key(image.names[callee])) not in known
			},
			key=_call_pair,
		)
	)


class _CodeMeasures(Struct):
	"""Frames measured from code, and the calls that code adds to the graph."""

	frames: tuple[Frame, ...]
	edges: tuple[CallEdge, ...]


def _register_callers(
	image: _Image, measure: CodeMeasure, sites: tuple[CallSite, ...], frames: tuple[StackUsage, ...]
) -> _CodeMeasures:
	framed = frozenset(frame_key(frame.function) for frame in frames)
	sites_by_start = Counter(
		normalized(site.caller_address, image.program.machine) for site in sites
	)
	owns = own_frames(
		measure,
		frozenset(
			start
			for start in sites_by_start
			if start in image.names and frame_key(image.names[start]) not in framed
		),
	)
	return _CodeMeasures(
		frames=tuple(
			OwnFrame(function=image.names[start], bytes=own.bytes) for start, own in owns.items()
		),
		edges=tuple(
			edge
			for start, own in sorted(owns.items())
			for edge in (
				*(
					CallEdge(caller=image.names[start], callee=INDIRECT_CALLEE)
					for _ in range(sites_by_start[start])
				),
				*_direct_edges(image, start, own),
			)
		),
	)


def _unrecorded_sites(
	image: _Image, sites: tuple[CallSite, ...], edges: tuple[CallEdge, ...]
) -> tuple[CallEdge, ...]:
	recorded = Counter(frame_key(edge.caller) for edge in edges if edge.callee == INDIRECT_CALLEE)
	return tuple(
		CallEdge(caller=caller, callee=INDIRECT_CALLEE)
		for caller, found in sorted(
			Counter(
				image.names[start]
				for site in sites
				if not calls_nothing(image.program, site)
				for start in (normalized(site.caller_address, image.program.machine),)
				if start in image.names
			).items()
		)
		for _ in range(found - recorded[frame_key(caller)])
	)


def _direct_edges(image: _Image, start: Address, own: OwnCode) -> tuple[CallEdge, ...]:
	return tuple(
		CallEdge(caller=image.names[start], callee=image.names[callee], kind=EdgeKind.BINARY)
		for callee in sorted(own.callees)
		if callee in image.names
	)


def _code_measures(
	image: _Image,
	measure: CodeMeasure,
	edges: tuple[CallEdge, ...],
	frames: tuple[Frame, ...],
	roots: frozenset[str],
	attempted: frozenset[str],
) -> _CodeMeasures:
	unframed = (
		frozenset(frame_key(node) for edge in edges for node in (edge.caller, edge.callee))
		| frozenset(map(frame_key, roots))
	) - ({frame_key(frame.function) for frame in frames} | {INDIRECT_CALLEE} | attempted)
	if not unframed:
		return _CodeMeasures(frames=(), edges=())
	starts = {address: name for address, name in image.names.items() if frame_key(name) in unframed}
	depths = code_depths(measure, frozenset(starts))
	owns = own_frames(measure, frozenset(starts) - depths.keys())
	measured = tuple(
		address
		for key in unframed
		for addresses in (
			frozenset(address for address in starts if frame_key(starts[address]) == key),
		)
		if addresses and addresses <= depths.keys() | owns.keys()
		for address in addresses
	)
	found = _CodeMeasures(
		frames=tuple(
			MeasuredFrame(function=starts[address], bytes=depths[address])
			if address in depths
			else OwnFrame(function=starts[address], bytes=owns[address].bytes)
			for address in measured
		),
		edges=tuple(
			edge
			for address in sorted(measured)
			if address not in depths
			for edge in _direct_edges(image, address, owns[address])
		),
	)
	following = _code_measures(
		image, measure, found.edges, (*frames, *found.frames), frozenset(), attempted | unframed
	)
	return _CodeMeasures(
		frames=(*found.frames, *following.frames), edges=(*found.edges, *following.edges)
	)


def _image_functions(artifacts: _Artifacts, image: _Image) -> frozenset[str]:
	match artifacts.membership:
		case Membership.LINKER:
			return artifacts.held
		case Membership.NAMES:
			return frozenset(
				map(frame_key, (*defined_function_names(image.elf), *image.names.values()))
			)
		case _ as unreachable:
			assert_never(unreachable)


def _thread_graph(
	image: _Image,
	model: RtosModel,
	edges: tuple[CallEdge, ...],
	frames: tuple[Frame, ...],
) -> tuple[tuple[CallEdge, ...], tuple[Frame, ...]]:
	entries_by_thread = {
		thread.name: (frame_key(stack_name(image.program, image.names, thread.entry)), started_by)
		for threads, started_by in (
			(model.threads, EdgeKind.THREAD),
			(model.system_threads, EdgeKind.SYSTEM_THREAD),
		)
		for thread in threads
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


class _Targets(Struct):
	"""The whole image's targets for the indirect calls, by caller key."""

	by_caller: Mapping[str, frozenset[str]]
	narrowed_by_caller: Mapping[str, Mapping[str, EdgeKind]]
	"""Reached only through a narrowing, with its edge."""
	fallback: frozenset[str]
	narrowed_by_field: tuple[NarrowedSpan, ...]
	narrow_by_signature: bool


def _own_thread_graphs(
	image: _Image,
	model: RtosModel,
	resolution: ProgramResolution,
	edges: tuple[CallEdge, ...],
	targets: _Targets,
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
					frame_key(stack_name(image.program, image.names, entries[thread])),
					own_targets(
						image.program,
						sites,
						resolution.assignments,
						image.names,
						targets.narrowed_by_field,
						narrow_by_signature=targets.narrow_by_signature,
						fallback=resolution.fallback,
					),
					targets.by_caller,
					targets.fallback,
					targets.narrowed_by_caller,
				)
				for thread, sites in resolution.threads.items()
			}
		case _ as unreachable:
			assert_never(unreachable)


def _in_image(reports: tuple[StackReport, ...], expansion: Expansion) -> tuple[StackReport, ...]:
	return tuple(
		report
		for report in reports
		if report.entry in expansion.image_functions
		or report.entry in expansion.threads
		or report.entry in expansion.hardware_roots
		or (
			expansion.rtos.interrupt_stack is not None
			and report.entry == expansion.rtos.interrupt_stack.name
		)
	)


def _with_interrupt_stack(
	expansion: Expansion | None, graph: StackGraph
) -> tuple[StackReport, ...]:
	return (*stack_reports(graph), *_interrupt_stacks(expansion, graph))


def _interrupt_stacks(expansion: Expansion | None, graph: StackGraph) -> tuple[StackReport, ...]:
	return (
		tuple(
			interrupt_stack_report(
				stack.name,
				_handled(graph, expansion.exceptions, RESET),
				tuple(
					_handled(graph, expansion.exceptions, number)
					for number in (HARD_FAULT, NMI)
					if number in expansion.exceptions
				),
				tuple(
					_handled(graph, expansion.exceptions, number)
					for number in sorted(expansion.exceptions)
					if number > HARD_FAULT
				),
				levels=expansion.priority_levels,
				exception_frame=expansion.rtos.exception_frame,
			)
			for stack in (expansion.rtos.interrupt_stack,)
			if stack is not None and RESET in expansion.exceptions
		)
		if expansion is not None
		else ()
	)


def _kconfig(elf: Path) -> Kconfig:
	return (
		kconfig((elf.parent / ".config").read_text(errors="replace"))
		if (elf.parent / ".config").is_file()
		else NO_KCONFIG
	)


def _handled(graph: StackGraph, exceptions: Mapping[int, str], number: int) -> Handled:
	return Handled(exception=number, report=entry_report(graph, exceptions[number]))


def _priority_levels(elf: Path, exceptions: Mapping[int, str]) -> PriorityLevels:
	match priority_levels(
		(elf.parent / "zephyr.dts").read_text(errors="replace")
		if (elf.parent / "zephyr.dts").is_file()
		else ""
	):
		case int() as count:
			return PriorityLevels(count=count, source=LevelSource.DEVICETREE)
		case None:
			return PriorityLevels(
				count=sum(number > HARD_FAULT for number in exceptions),
				source=LevelSource.VECTOR_TABLE,
			)
		case _ as unreachable:
			assert_never(unreachable)


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
	narrow_by_field: Annotated[bool, _NARROW_BY_FIELD] = False,
	assume_no_recursion: Annotated[tuple[str, ...], _ASSUME_NO_RECURSION] = (),
	assume_unwritten: Annotated[tuple[str, ...], _ASSUME_UNWRITTEN] = (),
	assume_frame: Annotated[tuple[str, ...], _ASSUME_FRAME] = (),
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
	image = _image(elf)
	artifacts = _artifacts(build_directory, image)
	expansion = (
		_framed(
			_expand_from_elf(
				artifacts,
				image,
				narrow_by_signature=narrow_by_signature,
				narrow_by_field=narrow_by_field,
				rtos=rtos,
				assume_unwritten=assume_unwritten,
			),
			assume_frame,
		)
		if image is not None
		else None
	)
	graph = stack_graph(
		expansion.expanded if expansion is not None else artifacts.edges,
		expansion.frames
		if expansion is not None
		else (
			*artifacts.frames,
			*_assumed_frames(artifacts.edges, artifacts.frames, assume_frame),
		),
		entry_edges=expansion.in_image_edges if expansion is not None else None,
		assumed_no_recursion=frozenset(assume_no_recursion),
		assumed_unwritten=expansion.assumed_unwritten.callers
		if expansion is not None
		else frozenset(),
		hardware_roots=expansion.hardware_roots if expansion is not None else frozenset(),
	)
	reports = _with_interrupt_stack(expansion, graph)
	own_graphs = _own_graphs(expansion, frozenset(assume_no_recursion))
	kept = (
		_in_image(thread_reports(reports, own_graphs, expansion.rtos), expansion)
		if expansion is not None
		else reports
	)
	shown = kept if path is None else _entry_report(kept, path)
	steps = None if path is None else _path_steps(shown[0], own_graphs.get(path, graph))
	if json:
		print(msgspec.json.format(msgspec.json.encode(_stack_document(shown, steps)).decode()))
		return
	if expansion is not None:
		print(
			f"resolved slots: {expansion.counts.resolved_slots} "
			f"| indirect call sites: {expansion.indirect_sites} "
			f"| not in the image: {_not_in_image(len(reports) - len(kept), expansion)} "
			f"| membership: {expansion.membership}"
			f"{' | narrowed by signature' if narrow_by_signature else ''}"
			f"{' | narrowed by field' if narrow_by_field else ''}"
			f"{f' | assumed no recursion: {", ".join(sorted(assume_no_recursion))}' if assume_no_recursion else ''}"
			f"{f' | assumed unwritten: {", ".join(sorted(assume_unwritten))}' if assume_unwritten else ''}"
			f"{f' | yet tracked stores write: {", ".join(expansion.assumed_unwritten.contradicted)}' if expansion.assumed_unwritten.contradicted else ''}"
			f"{f' | assumed frames: {", ".join(sorted(assume_frame))}' if assume_frame else ''}"
			f"{f' | rtos: {expansion.rtos.name}' if expansion.rtos.evidence else ''}"
		)
	for report in shown:
		print(_render(report))
	for step in steps or ():
		print(_render_step(step))
	for line in _thread_stack_steps(shown, steps):
		print(line)
	for line in _nesting_steps(shown, steps):
		print(line)


def _not_in_image(names_dropped: int, expansion: Expansion) -> str:
	match expansion.membership:
		case Membership.NAMES:
			return str(names_dropped)
		case Membership.LINKER:
			return (
				f"{sum(map(len, expansion.dropped.values()))} "
				f"(discarded: {len(expansion.dropped[Linkage.DISCARDED])}, "
				f"never linked: {len(expansion.dropped[Linkage.NEVER_LINKED])})"
			)
		case _ as unreachable:
			assert_never(unreachable)


def _framed(expansion: Expansion, stated: tuple[str, ...]) -> Expansion:
	return replace(
		expansion,
		frames=(
			*expansion.frames,
			*_assumed_frames(expansion.expanded, expansion.frames, stated),
		),
	)


def _assumed_frames(
	edges: tuple[CallEdge, ...], frames: tuple[Frame, ...], stated: tuple[str, ...]
) -> tuple[AssumedFrame, ...]:
	return tuple(_assumed_frame(edges, frames, statement) for statement in stated)


def _assumed_frame(
	edges: tuple[CallEdge, ...], frames: tuple[Frame, ...], statement: str
) -> AssumedFrame:
	name, separator, size = statement.partition("=")
	if not separator or not size.isdigit():
		raise ValueError(f"--assume-frame: {statement} is not FUNCTION=BYTES")
	if any(frame_key(frame.function) == frame_key(name) for frame in frames):
		raise ValueError(f"--assume-frame: {name} already has a frame")
	if not any(
		frame_key(node) == frame_key(name) for edge in edges for node in (edge.caller, edge.callee)
	):
		raise ValueError(f"--assume-frame: no function named {name} is in the call graph")
	return AssumedFrame(function=name, bytes=int(size))


def _own_graphs(
	expansion: Expansion | None, assumed_no_recursion: frozenset[str]
) -> Mapping[str, StackGraph]:
	return (
		{
			thread: stack_graph(
				edges,
				expansion.frames,
				assumed_no_recursion=assumed_no_recursion,
				assumed_unwritten=expansion.assumed_unwritten.callers
				| expansion.assumed_unwritten.thread_callers.get(thread, frozenset()),
			)
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
			StackEntryReport(
				entry=report.entry,
				bound=stack_bound_report(report.bound),
				nesting=_nesting_document(report.nesting),
				stack_bytes=report.stack_size,
				margin_bytes=_margin(report),
			)
			for report in shown
		)
		if steps is None
		else StackPathReport(
			entry=shown[0].entry,
			bound=stack_bound_report(shown[0].bound),
			path=tuple(_step_report(step) for step in steps),
			nesting=_nesting_document(shown[0].nesting),
			stack_bytes=shown[0].stack_size,
			margin_bytes=_margin(shown[0]),
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
	narrow_by_field: Annotated[bool, _NARROW_BY_FIELD] = False,
	assume_no_recursion: Annotated[tuple[str, ...], _ASSUME_NO_RECURSION] = (),
	assume_unwritten: Annotated[tuple[str, ...], _ASSUME_UNWRITTEN] = (),
	assume_frame: Annotated[tuple[str, ...], _ASSUME_FRAME] = (),
	rtos: Annotated[RtosChoice, _RTOS] = RtosChoice.AUTO,
) -> None:
	"""Print a JSON summary combining resolution rates and worst-case stack depth."""
	expansion = _framed(
		elf_expansion(
			build_directory,
			elf,
			narrow_by_signature=narrow_by_signature,
			narrow_by_field=narrow_by_field,
			rtos=rtos,
			assume_unwritten=assume_unwritten,
		),
		assume_frame,
	)
	reports = _with_interrupt_stack(
		expansion,
		stack_graph(
			expansion.expanded,
			expansion.frames,
			entry_edges=expansion.in_image_edges,
			assumed_no_recursion=frozenset(assume_no_recursion),
			assumed_unwritten=expansion.assumed_unwritten.callers,
			hardware_roots=expansion.hardware_roots,
		),
	)
	kept = _in_image(
		thread_reports(
			reports,
			_own_graphs(expansion, frozenset(assume_no_recursion)),
			expansion.rtos,
		),
		expansion,
	)
	worst = next(iter(kept), StackReport(entry="", bound=Bounded(bytes=0)))
	report = AnalysisSummary(
		resolved_slots=expansion.counts.resolved_slots,
		total_slots=expansion.counts.total_slots,
		unresolved_slots=len(expansion.unresolved),
		resolved_targets=expansion.counts.resolved_targets,
		indirect_call_sites=expansion.indirect_sites,
		total_functions=len(expansion.program.functions),
		entry_points=len(kept),
		discarded_entry_points=len(reports)
		- len(kept)
		+ len(expansion.dropped.get(Linkage.DISCARDED, frozenset())),
		worst_case_entry=worst.entry,
		worst_case=stack_bound_report(worst.bound),
		rtos=expansion.rtos.name,
		membership=expansion.membership,
		never_linked_entry_points=(
			len(expansion.dropped[Linkage.NEVER_LINKED])
			if expansion.membership is Membership.LINKER
			else None
		),
		phantom_libcalls=(
			tuple(sorted(expansion.phantom_libcalls))
			if expansion.membership is Membership.LINKER
			else None
		),
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
