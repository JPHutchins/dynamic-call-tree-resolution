# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Stack bounds for entries, threads and interrupts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, assert_never

from salix import Struct, replace

from dynamic_call_tree_resolution.model import BARE_METAL
from dynamic_call_tree_resolution.stack_analysis.graph import (
	own_reasons,
	stack_graph,
)
from dynamic_call_tree_resolution.stack_analysis.values import (
	Bounded,
	Frame,
	FunctionReason,
	Handled,
	Nested,
	Nesting,
	PriorityLevels,
	Reason,
	Reasons,
	StackGraph,
	StackReport,
	Unbounded,
	frame_key,
)

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping

	from dynamic_call_tree_resolution.callgraph import CallEdge
	from dynamic_call_tree_resolution.model import RtosModel


def worst_case_depths(
	edges: Iterable[CallEdge],
	frames: Iterable[Frame],
	*,
	entry_edges: Iterable[CallEdge] | None = None,
) -> tuple[StackReport, ...]:
	return stack_reports(stack_graph(edges, frames, entry_edges=entry_edges))


def stack_reports(graph: StackGraph) -> tuple[StackReport, ...]:
	return tuple(sorted((root_report(graph, root) for root in graph.roots), key=report_order))


def thread_reports(
	reports: Iterable[StackReport],
	own_graphs: Mapping[str, StackGraph],
	rtos: RtosModel = BARE_METAL,
) -> tuple[StackReport, ...]:
	return tuple(
		sorted(
			(
				replace(
					_thread_report(report, own_graphs, rtos), stack_size=sizes.get(report.entry)
				)
				for sizes in (_stack_sizes(rtos),)
				for report in reports
			),
			key=report_order,
		)
	)


def _stack_sizes(rtos: RtosModel) -> Mapping[str, int]:
	return {
		stack.name: stack.stack_size
		for stack in (*rtos.threads, *rtos.system_threads, rtos.interrupt_stack)
		if stack is not None and stack.stack_size is not None
	}


def _thread_report(
	report: StackReport, own_graphs: Mapping[str, StackGraph], rtos: RtosModel
) -> StackReport:
	match own_graphs.get(report.entry):
		case StackGraph() as own:
			return _on_thread_stack(root_report(own, report.entry), rtos)
		case None if any(thread.name == report.entry for thread in rtos.system_threads):
			return _on_thread_stack(report, rtos)
		case None:
			return report
		case _ as unreachable:
			assert_never(unreachable)


def _on_thread_stack(report: StackReport, rtos: RtosModel) -> StackReport:
	match report.bound:
		case Bounded(bytes=depth):
			return replace(
				report,
				bound=replace(
					report.bound,
					bytes=depth + rtos.stack_reservation + rtos.exception_frame,
					stack_reservation=rtos.stack_reservation,
					exception_frame=rtos.exception_frame,
				),
			)
		case Unbounded(at_least=at_least):
			return replace(
				report,
				bound=replace(
					report.bound,
					at_least=at_least + rtos.stack_reservation + rtos.exception_frame,
					stack_reservation=rtos.stack_reservation,
					exception_frame=rtos.exception_frame,
				),
			)
		case _ as unreachable:
			assert_never(unreachable)


def entry_report(graph: StackGraph, entry: str) -> StackReport:
	return min(
		(
			root_report(graph, node)
			for node in graph.depths.keys() | graph.frame_by_name.keys()
			if frame_key(node) == entry
		),
		key=report_order,
		default=root_report(graph, entry),
	)


def root_report(graph: StackGraph, root: str) -> StackReport:
	return StackReport(
		entry=frame_key(root), bound=_bound(_root_depth(graph, root), _root_reasons(graph, root))
	)


def _root_depth(graph: StackGraph, root: str) -> int:
	return graph.depths[root] if root in graph.depths else frame_size(graph, root)


def _root_reasons(graph: StackGraph, root: str) -> Reasons:
	return (
		graph.reasons[root]
		if root in graph.reasons
		else own_reasons(
			root, graph.adjacency, graph.frame_by_name, graph.assumed_no_recursion, graph.narrowings
		)
	)


class ReportOrder(Struct, order=True):
	"""Where a report sorts among reports."""

	bounded: bool
	negated_depth: int


def report_order(report: StackReport) -> ReportOrder:
	match report.bound:
		case Unbounded(at_least=at_least):
			return ReportOrder(bounded=False, negated_depth=-at_least)
		case Bounded(bytes=depth):
			return ReportOrder(bounded=True, negated_depth=-depth)
		case _ as unreachable:
			assert_never(unreachable)


_NOTED: Final = frozenset(
	{
		Reason.MEASURED,
		Reason.ASSUMED_NO_RECURSION,
		Reason.NARROWED_BY_FIELD,
		Reason.NARROWED_BY_SIGNATURE,
		Reason.ASSUMED_UNWRITTEN,
		Reason.ASSUMED_FRAME,
	}
)


def _bound(depth: int, reasons: Reasons) -> Bounded | Unbounded:
	return (
		Unbounded(
			at_least=depth,
			recursion=_names(reasons, Reason.RECURSION),
			unmeasured=_names(reasons, Reason.UNMEASURED),
			dynamic=_names(reasons, Reason.DYNAMIC),
			unresolved=_names(reasons, Reason.UNRESOLVED),
			measured=_names(reasons, Reason.MEASURED),
			assumed_no_recursion=_names(reasons, Reason.ASSUMED_NO_RECURSION),
			narrowed_by_field=_names(reasons, Reason.NARROWED_BY_FIELD),
			narrowed_by_signature=_names(reasons, Reason.NARROWED_BY_SIGNATURE),
			assumed_unwritten=_names(reasons, Reason.ASSUMED_UNWRITTEN),
			assumed_frames=_names(reasons, Reason.ASSUMED_FRAME),
		)
		if any(function_reason.reason not in _NOTED for function_reason in reasons)
		else Bounded(
			bytes=depth,
			measured=_names(reasons, Reason.MEASURED),
			assumed_no_recursion=_names(reasons, Reason.ASSUMED_NO_RECURSION),
			narrowed_by_field=_names(reasons, Reason.NARROWED_BY_FIELD),
			narrowed_by_signature=_names(reasons, Reason.NARROWED_BY_SIGNATURE),
			assumed_unwritten=_names(reasons, Reason.ASSUMED_UNWRITTEN),
			assumed_frames=_names(reasons, Reason.ASSUMED_FRAME),
		)
	)


def _names(reasons: Reasons, reason: Reason) -> frozenset[str]:
	return frozenset(
		function_reason.function for function_reason in reasons if function_reason.reason is reason
	)


def frame_size(graph: StackGraph, node: str) -> int:
	frame = graph.frame_by_name.get(frame_key(node))
	return frame.bytes if frame is not None else 0


def interrupt_stack_report(
	name: str,
	base: Handled,
	fixed: tuple[Handled, ...],
	configurable: tuple[Handled, ...],
	*,
	levels: PriorityLevels,
	exception_frame: int,
) -> StackReport:
	chain = (*sorted(configurable, key=_deepest_first)[: levels.count], *fixed)
	return StackReport(
		entry=name,
		bound=_bound(
			_bound_depth(base.report.bound)
			+ sum(exception_frame + _bound_depth(handled.report.bound) for handled in chain),
			frozenset[FunctionReason]().union(
				*(_bound_reasons(handled.report.bound) for handled in (base, *fixed, *configurable))
			),
		),
		nesting=Nesting(
			base=_nested(base),
			chain=tuple(map(_nested, chain)),
			levels=levels,
			exception_frame=exception_frame,
		),
	)


def _nested(handled: Handled) -> Nested:
	return Nested(
		exception=handled.exception,
		handler=handled.report.entry,
		depth=_bound_depth(handled.report.bound),
	)


def _deepest_first(handled: Handled) -> int:
	return -_bound_depth(handled.report.bound)


def _bound_depth(bound: Bounded | Unbounded) -> int:
	match bound:
		case Bounded(bytes=depth):
			return depth
		case Unbounded(at_least=at_least):
			return at_least
		case _ as unreachable:
			assert_never(unreachable)


class _ReasonFunctions(Struct):
	"""A reason a bound carries, and every function it comes from."""

	reason: Reason
	functions: frozenset[str]


def _bound_reasons(bound: Bounded | Unbounded) -> Reasons:
	return frozenset(
		FunctionReason(reason=reason_functions.reason, function=function)
		for reason_functions in _reason_sets(bound)
		for function in reason_functions.functions
	)


def _reason_sets(bound: Bounded | Unbounded) -> tuple[_ReasonFunctions, ...]:
	match bound:
		case Bounded():
			return _noted_sets(bound)
		case Unbounded():
			return (
				_ReasonFunctions(reason=Reason.RECURSION, functions=bound.recursion),
				_ReasonFunctions(reason=Reason.UNMEASURED, functions=bound.unmeasured),
				_ReasonFunctions(reason=Reason.DYNAMIC, functions=bound.dynamic),
				_ReasonFunctions(reason=Reason.UNRESOLVED, functions=bound.unresolved),
				*_noted_sets(bound),
			)
		case _ as unreachable:
			assert_never(unreachable)


def _noted_sets(bound: Bounded | Unbounded) -> tuple[_ReasonFunctions, ...]:
	return (
		_ReasonFunctions(reason=Reason.MEASURED, functions=bound.measured),
		_ReasonFunctions(reason=Reason.ASSUMED_NO_RECURSION, functions=bound.assumed_no_recursion),
		_ReasonFunctions(reason=Reason.NARROWED_BY_FIELD, functions=bound.narrowed_by_field),
		_ReasonFunctions(
			reason=Reason.NARROWED_BY_SIGNATURE, functions=bound.narrowed_by_signature
		),
		_ReasonFunctions(reason=Reason.ASSUMED_UNWRITTEN, functions=bound.assumed_unwritten),
		_ReasonFunctions(reason=Reason.ASSUMED_FRAME, functions=bound.assumed_frames),
	)
