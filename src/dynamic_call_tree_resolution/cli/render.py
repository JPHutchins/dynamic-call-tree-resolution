# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Rendering reports, stack rows and notes as text."""

import sys
from itertools import accumulate
from typing import TYPE_CHECKING, assert_never

from salix import Struct

from dynamic_call_tree_resolution.cli.app import (
	oversized,
)
from dynamic_call_tree_resolution.loader import (
	elf_machine,
)
from dynamic_call_tree_resolution.model import Machine, Residue
from dynamic_call_tree_resolution.report import (
	NestingReport,
	PathStepReport,
	RtosReport,
	SignatureReport,
	nesting_report,
)
from dynamic_call_tree_resolution.stack_analysis import (
	Bounded,
	Nesting,
	PathStep,
	Reason,
	StackGraph,
	StackReport,
	Unbounded,
	deepest_path,
)

if TYPE_CHECKING:
	from pathlib import Path


def _skip_reason(path: Path) -> str | None:
	if oversized(path):
		return f"{path.stat().st_size // 1024 // 1024} MiB; too large"
	if (machine := elf_machine(path)) not in Machine:
		return f"{machine}; unsupported machine"
	return None


def render(report: StackReport) -> str:
	match report.bound:
		case Bounded(bytes=depth) as bound:
			notes = _notes(
				(
					_Count(reason=Reason.MEASURED, functions=bound.measured),
					_Count(
						reason=Reason.ASSUMED_NO_RECURSION, functions=bound.assumed_no_recursion
					),
					_Count(reason=Reason.NARROWED_BY_FIELD, functions=bound.narrowed_by_field),
					_Count(
						reason=Reason.NARROWED_BY_SIGNATURE, functions=bound.narrowed_by_signature
					),
					_Count(reason=Reason.ASSUMED_UNWRITTEN, functions=bound.assumed_unwritten),
					_Count(reason=Reason.ASSUMED_FRAME, functions=bound.assumed_frames),
				),
				_thread_stack_additions(bound),
				(*_nesting_notes(report.nesting), *_stack_notes(report)),
			)
			return f"{report.entry}: {depth} bytes" + (f" ({notes})" if notes else "")
		case Unbounded() as bound:
			notes = _notes(
				(
					_Count(reason=Reason.RECURSION, functions=bound.recursion),
					_Count(reason=Reason.UNMEASURED, functions=bound.unmeasured),
					_Count(reason=Reason.MEASURED, functions=bound.measured),
					_Count(
						reason=Reason.ASSUMED_NO_RECURSION, functions=bound.assumed_no_recursion
					),
					_Count(reason=Reason.NARROWED_BY_FIELD, functions=bound.narrowed_by_field),
					_Count(
						reason=Reason.NARROWED_BY_SIGNATURE, functions=bound.narrowed_by_signature
					),
					_Count(reason=Reason.ASSUMED_UNWRITTEN, functions=bound.assumed_unwritten),
					_Count(reason=Reason.ASSUMED_FRAME, functions=bound.assumed_frames),
					_Count(reason=Reason.DYNAMIC, functions=bound.dynamic),
					_Count(reason=Reason.UNRESOLVED, functions=bound.unresolved),
				),
				_thread_stack_additions(bound),
				(*_nesting_notes(report.nesting), *_stack_notes(report)),
			)
			return f"{report.entry}: unbounded, at least {bound.at_least} bytes ({notes})"
		case _ as unreachable:
			assert_never(unreachable)


class _Count(Struct):
	"""The functions a report names for one reason."""

	reason: Reason
	functions: frozenset[str]


class _Addition(Struct):
	"""Bytes a thread's stack holds beyond its deepest path."""

	label: str
	added: int


def _notes(
	counts: tuple[_Count, ...],
	additions: tuple[_Addition, ...],
	nesting: tuple[str, ...],
) -> str:
	return ", ".join(
		(
			*(f"{count.reason}: {len(count.functions)}" for count in counts if count.functions),
			*(
				f"{addition.label}: {addition.added} bytes"
				for addition in additions
				if addition.added
			),
			*nesting,
		)
	)


def _thread_stack_additions(bound: Bounded | Unbounded) -> tuple[_Addition, ...]:
	return (
		_Addition(label="stack reservation", added=bound.stack_reservation),
		_Addition(label="exception frame", added=bound.exception_frame),
	)


def thread_stack_steps(
	shown: tuple[StackReport, ...], steps: tuple[PathStep, ...] | None
) -> tuple[str, ...]:
	return tuple(
		f"({addition.label}) +{addition.added} = {steps[-1].cumulative + total} bytes"
		for report in shown[:1]
		if steps
		for additions in (_thread_stack_additions(report.bound),)
		for addition, total in zip(
			additions, accumulate(addition.added for addition in additions), strict=True
		)
		if addition.added
	)


def path_steps(report: StackReport, graph: StackGraph) -> tuple[PathStep, ...]:
	return deepest_path(graph, report.entry) if report.nesting is None else ()


def nesting_steps(
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


def nesting_document(nesting: Nesting | None) -> NestingReport | None:
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
		*(f"margin: {margin} bytes" for margin in (stack_margin(report),) if margin is not None),
	)


def stack_margin(report: StackReport) -> int | None:
	match report.bound, report.stack_size:
		case Bounded(bytes=depth), int() as stack_size:
			return stack_size - depth
		case ((Bounded() | Unbounded()), (int() | None)):
			return None
		case _ as unreachable:
			assert_never(unreachable)


def signature_text(signature: SignatureReport) -> str:
	return f"{signature.return_type} ({', '.join(signature.parameters)})"


def residue_label(residue: Residue) -> str:
	match residue:
		case Residue.ROM_NON_FUNCTION:
			return "<not a function>"
		case Residue.RAM_NULL | Residue.RAM_UNINITIALIZED | Residue.RAM_INITIALIZED:
			return "<unresolved>"
		case _ as unreachable:
			assert_never(unreachable)


def render_step(step: PathStep) -> str:
	return (
		f"{step.function} +{step.frame} = {step.cumulative} bytes"
		f"{' via ' + ', '.join(sorted(step.edge)) if step.edge else ''}"
		f"{' (' + ', '.join(sorted(step.flags)) + ')' if step.flags else ''}"
	)


def step_report(step: PathStep) -> PathStepReport:
	return PathStepReport(
		function=step.function,
		frame_bytes=step.frame,
		cumulative_bytes=step.cumulative,
		edge=tuple(sorted(step.edge)),
		flags=tuple(sorted(step.flags)),
	)


def rtos_lines(report: RtosReport) -> tuple[str, ...]:
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


def keep(path: Path) -> bool:
	match _skip_reason(path):
		case None:
			return True
		case str() as reason:
			print(f"warning: skipping {path} ({reason})", file=sys.stderr)
			return False
		case _ as unreachable:
			assert_never(unreachable)
