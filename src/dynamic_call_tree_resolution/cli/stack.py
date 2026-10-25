# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The ``stack`` command."""

from pathlib import Path  # noqa: TC003  # cyclopts evaluates Annotated[Path, ...] at runtime
from typing import TYPE_CHECKING, Annotated, Final, assert_never

import msgspec
from cyclopts import Parameter
from salix import replace

from dynamic_call_tree_resolution.cli.app import (
	ASSUME_FRAME,
	ASSUME_NO_RECURSION,
	ASSUME_UNWRITTEN,
	NARROW_BY_FIELD,
	NARROW_BY_SIGNATURE,
	RTOS,
	app,
)
from dynamic_call_tree_resolution.cli.artifacts import (
	Expansion,
	build_artifacts,
	optional_image,
)
from dynamic_call_tree_resolution.cli.expansion import (
	expand_from_elf,
)
from dynamic_call_tree_resolution.cli.measures import SYSTEM_CALL_NODE
from dynamic_call_tree_resolution.cli.render import (
	nesting_document,
	nesting_steps,
	path_steps,
	render,
	render_step,
	stack_margin,
	step_report,
	thread_stack_steps,
)
from dynamic_call_tree_resolution.linker import (
	Linkage,
	Membership,
)
from dynamic_call_tree_resolution.report import (
	StackEntryReport,
	StackPathReport,
	stack_bound_report,
)
from dynamic_call_tree_resolution.rtos import RtosChoice
from dynamic_call_tree_resolution.stack_analysis import (
	AssumedFrame,
	Frame,
	Handled,
	LevelSource,
	PathStep,
	PriorityLevels,
	StackGraph,
	StackReport,
	entry_report,
	frame_key,
	interrupt_stack_report,
	report_order,
	stack_graph,
	stack_reports,
	thread_reports,
)
from dynamic_call_tree_resolution.vsa.vectors import (
	HARD_FAULT,
	NMI,
	RESET,
)

if TYPE_CHECKING:
	from collections.abc import Mapping

	from dynamic_call_tree_resolution.callgraph import CallEdge


def in_image(reports: tuple[StackReport, ...], expansion: Expansion) -> tuple[StackReport, ...]:
	return tuple(
		report
		for report in reports
		if report.entry in expansion.image_functions
		or report.entry in expansion.threads
		or report.entry in expansion.hardware_roots
		or any(
			stack is not None and report.entry == stack.name
			for stack in (
				expansion.rtos.interrupt_stack,
				expansion.rtos.privileged_stack,
				*expansion.rtos.mode_stacks,
				expansion.rtos.handler_stack,
			)
		)
	)


def with_interrupt_stack(expansion: Expansion | None, graph: StackGraph) -> tuple[StackReport, ...]:
	return (
		*stack_reports(graph),
		*_interrupt_stacks(expansion, graph),
		*_handler_stacks(expansion, graph),
	)


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


def _handled(graph: StackGraph, exceptions: Mapping[int, str], number: int) -> Handled:
	return Handled(exception=number, report=entry_report(graph, exceptions[number]))


def _handler_stacks(expansion: Expansion | None, graph: StackGraph) -> tuple[StackReport, ...]:
	return (
		tuple(
			interrupt_stack_report(
				stack.name,
				Handled(exception=_SVC, report=entry_report(graph, SYSTEM_CALL_NODE)),
				tuple(
					_wrapped(graph, number, handler, names.wrapper_calls)
					for number, handler in names.spurious.items()
				),
				tuple(
					_wrapped(graph, number, handler, names.wrapper_calls)
					for number, handler in names.interrupts.items()
				),
				levels=PriorityLevels(
					count=len(names.interrupts), source=LevelSource.INTERRUPT_TABLE
				),
				exception_frame=stack.interrupt_frame,
			)
			for stack in (expansion.rtos.handler_stack,)
			for names in (expansion.handlers,)
			if stack is not None and names is not None
		)
		if expansion is not None
		else ()
	)


_SVC: Final = 2


def _wrapped(graph: StackGraph, number: int, handler: str, calls: tuple[str, ...]) -> Handled:
	return Handled(
		exception=number,
		report=min((entry_report(graph, name) for name in (handler, *calls)), key=report_order),
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
	narrow_by_signature: Annotated[bool, NARROW_BY_SIGNATURE] = False,
	narrow_by_field: Annotated[bool, NARROW_BY_FIELD] = False,
	assume_no_recursion: Annotated[tuple[str, ...], ASSUME_NO_RECURSION] = (),
	assume_unwritten: Annotated[tuple[str, ...], ASSUME_UNWRITTEN] = (),
	assume_frame: Annotated[tuple[str, ...], ASSUME_FRAME] = (),
	rtos: Annotated[RtosChoice, RTOS] = RtosChoice.AUTO,
	path: Annotated[
		str | None,
		Parameter(
			help="print only this entry, with its deepest path: each function's frame and "
			"cumulative bytes, the edge it is called through, and what makes it unbounded"
		),
	] = None,
) -> None:
	"""Print worst-case stack depths of a build directory (.su and .ci artifacts)."""
	image = optional_image(elf)
	artifacts = build_artifacts(build_directory, image)
	expansion = (
		with_assumed_frames(
			expand_from_elf(
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
	reports = with_interrupt_stack(expansion, graph)
	own_graphs = own_stack_graphs(expansion, frozenset(assume_no_recursion))
	kept = (
		in_image(thread_reports(reports, own_graphs, expansion.rtos), expansion)
		if expansion is not None
		else reports
	)
	shown = kept if path is None else _entry_report(kept, path)
	steps = None if path is None else path_steps(shown[0], own_graphs.get(path, graph))
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
		print(render(report))
	for step in steps or ():
		print(render_step(step))
	for line in thread_stack_steps(shown, steps):
		print(line)
	for line in nesting_steps(shown, steps):
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


def with_assumed_frames(expansion: Expansion, stated: tuple[str, ...]) -> Expansion:
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


def own_stack_graphs(
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
				nesting=nesting_document(report.nesting),
				stack_bytes=report.stack_size,
				margin_bytes=stack_margin(report),
			)
			for report in shown
		)
		if steps is None
		else StackPathReport(
			entry=shown[0].entry,
			bound=stack_bound_report(shown[0].bound),
			path=tuple(step_report(step) for step in steps),
			nesting=nesting_document(shown[0].nesting),
			stack_bytes=shown[0].stack_size,
			margin_bytes=stack_margin(shown[0]),
		)
	)


def _entry_report(reports: tuple[StackReport, ...], entry: str) -> tuple[StackReport, ...]:
	report = next((report for report in reports if report.entry == entry), None)
	if report is None:
		raise ValueError(f"{entry} is not a stack entry")
	return (report,)
