# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Stack bounds, paths and the summary, as reports."""

from typing import assert_never

from msgspec import Struct

from dynamic_call_tree_resolution.callgraph import EdgeKind
from dynamic_call_tree_resolution.linker import Membership
from dynamic_call_tree_resolution.model import Uncounted
from dynamic_call_tree_resolution.stack_analysis import (
	Bounded,
	LevelSource,
	Nested,
	Nesting,
	Reason,
	Unbounded,
)


class BoundedStack(Struct, tag="bounded", tag_field="kind"):
	"""A stack depth that bounds every path of the call graph."""

	bytes: int
	measured: tuple[str, ...] = ()
	"""Reachable functions whose frames come from their code."""
	stack_reservation_bytes: int = 0
	"""The top of a thread's stack the RTOS model takes before the thread runs, included in
	``bytes``."""
	exception_frame_bytes: int = 0
	"""The interrupt frames the RTOS model adds to a thread's depth, included in ``bytes``."""
	exception_frames: int = 0
	"""How many interrupts' frames ``exception_frame_bytes`` holds."""
	stack_sentinel_bytes: int = 0
	"""The bottom of a thread's stack the RTOS model takes to catch an overflow, included in
	``bytes``."""
	assumed_no_recursion: tuple[str, ...] = ()
	"""Reachable functions assumed never to call themselves."""
	narrowed_by_field: tuple[str, ...] = ()
	"""Reachable callers whose indirect call ``--narrow-by-field`` narrowed."""
	narrowed_by_signature: tuple[str, ...] = ()
	"""Reachable callers whose indirect call ``--narrow-by-signature`` narrowed."""
	assumed_unwritten: tuple[str, ...] = ()
	"""Reachable callers whose indirect calls rest on ``--assume-unwritten``."""
	assumed_frames: tuple[str, ...] = ()
	"""Reachable functions whose frames ``--assume-frame`` states."""


class UnboundedStack(Struct, tag="unbounded", tag_field="kind"):
	"""The JSON form of ``Unbounded``."""

	at_least_bytes: int
	recursion: tuple[str, ...]
	unmeasured: tuple[str, ...]
	dynamic: tuple[str, ...]
	unresolved: tuple[str, ...]
	measured: tuple[str, ...] = ()
	"""Reachable functions whose frames come from their code."""
	stack_reservation_bytes: int = 0
	"""The top of a thread's stack the RTOS model takes before the thread runs, included in
	``at_least_bytes``."""
	exception_frame_bytes: int = 0
	"""The interrupt frames the RTOS model adds to a thread's depth, included in
	``at_least_bytes``."""
	exception_frames: int = 0
	"""How many interrupts' frames ``exception_frame_bytes`` holds."""
	stack_sentinel_bytes: int = 0
	"""The bottom of a thread's stack the RTOS model takes to catch an overflow, included in
	``at_least_bytes``."""
	assumed_no_recursion: tuple[str, ...] = ()
	"""Reachable functions assumed never to call themselves."""
	narrowed_by_field: tuple[str, ...] = ()
	"""Reachable callers whose indirect call ``--narrow-by-field`` narrowed."""
	narrowed_by_signature: tuple[str, ...] = ()
	"""Reachable callers whose indirect call ``--narrow-by-signature`` narrowed."""
	assumed_unwritten: tuple[str, ...] = ()
	"""Reachable callers whose indirect calls rest on ``--assume-unwritten``."""
	assumed_frames: tuple[str, ...] = ()
	"""Reachable functions whose frames ``--assume-frame`` states."""
	not_counted: tuple[Uncounted, ...] = ()
	"""What the RTOS adds to a thread's stack that its model leaves out of ``at_least_bytes``."""


def stack_bound_report(bound: Bounded | Unbounded) -> BoundedStack | UnboundedStack:
	match bound:
		case Bounded(bytes=depth, measured=measured, exception_frame=exception_frame):
			return BoundedStack(
				bytes=depth,
				measured=tuple(sorted(measured)),
				stack_reservation_bytes=bound.stack_reservation,
				exception_frame_bytes=exception_frame,
				exception_frames=bound.exception_frames,
				stack_sentinel_bytes=bound.stack_sentinel,
				assumed_no_recursion=tuple(sorted(bound.assumed_no_recursion)),
				narrowed_by_field=tuple(sorted(bound.narrowed_by_field)),
				narrowed_by_signature=tuple(sorted(bound.narrowed_by_signature)),
				assumed_unwritten=tuple(sorted(bound.assumed_unwritten)),
				assumed_frames=tuple(sorted(bound.assumed_frames)),
			)
		case Unbounded():
			return UnboundedStack(
				at_least_bytes=bound.at_least,
				recursion=tuple(sorted(bound.recursion)),
				unmeasured=tuple(sorted(bound.unmeasured)),
				dynamic=tuple(sorted(bound.dynamic)),
				unresolved=tuple(sorted(bound.unresolved)),
				measured=tuple(sorted(bound.measured)),
				stack_reservation_bytes=bound.stack_reservation,
				exception_frame_bytes=bound.exception_frame,
				exception_frames=bound.exception_frames,
				stack_sentinel_bytes=bound.stack_sentinel,
				assumed_no_recursion=tuple(sorted(bound.assumed_no_recursion)),
				narrowed_by_field=tuple(sorted(bound.narrowed_by_field)),
				narrowed_by_signature=tuple(sorted(bound.narrowed_by_signature)),
				assumed_unwritten=tuple(sorted(bound.assumed_unwritten)),
				assumed_frames=tuple(sorted(bound.assumed_frames)),
				not_counted=tuple(sorted(bound.not_counted)),
			)
		case _ as unreachable:
			assert_never(unreachable)


class NestedReport(Struct):
	"""One exception an interrupt stack's depth stacks."""

	exception: int
	handler: str
	depth_bytes: int
	"""The handler's depth, or its lower bound."""


class NestingReport(Struct):
	"""How an interrupt stack's depth stacks exceptions on the code that starts on it."""

	base: NestedReport
	chain: tuple[NestedReport, ...]
	"""The nested exceptions' handlers, in the order the depth adds them."""
	priority_levels: int
	"""The most configurable-priority exceptions one chain holds."""
	priority_levels_from: LevelSource
	exception_frame_bytes: int
	"""What each nested exception's hardware frame adds."""


def nesting_report(nesting: Nesting) -> NestingReport:
	return NestingReport(
		base=_nested_report(nesting.base),
		chain=tuple(map(_nested_report, nesting.chain)),
		priority_levels=nesting.levels.count,
		priority_levels_from=nesting.levels.source,
		exception_frame_bytes=nesting.exception_frame,
	)


def _nested_report(nested: Nested) -> NestedReport:
	return NestedReport(
		exception=nested.exception, handler=nested.handler, depth_bytes=nested.depth
	)


class StackEntryReport(Struct, omit_defaults=True):
	"""One row of ``stack --json``."""

	entry: str
	bound: BoundedStack | UnboundedStack
	nesting: NestingReport | None = None
	"""How an interrupt stack's row stacks its exceptions."""
	stack_bytes: int | None = None
	"""The size of the stack the entry runs on, as the RTOS model declares it."""
	margin_bytes: int | None = None
	"""``stack_bytes`` less a bounded depth; negative when the depth exceeds the stack."""


class PathStepReport(Struct):
	"""One function on a deepest path, rendered for consumers."""

	function: str
	frame_bytes: int
	cumulative_bytes: int
	edge: tuple[EdgeKind, ...]
	flags: tuple[Reason, ...]


class StackPathReport(Struct, omit_defaults=True):
	"""The output of ``stack --path --json``."""

	entry: str
	bound: BoundedStack | UnboundedStack
	path: tuple[PathStepReport, ...]
	nesting: NestingReport | None = None
	"""How an interrupt stack's row stacks its exceptions, in place of a call path."""
	stack_bytes: int | None = None
	"""The size of the stack the entry runs on, as the RTOS model declares it."""
	margin_bytes: int | None = None
	"""``stack_bytes`` less a bounded depth; negative when the depth exceeds the stack."""


class AnalysisSummary(Struct):
	"""Cross-cutting resolution and stack summary for CI reporting."""

	resolved_slots: int
	total_slots: int
	unresolved_slots: int
	resolved_targets: int
	indirect_call_sites: int
	total_functions: int
	entry_points: int
	discarded_entry_points: int
	worst_case_entry: str
	worst_case: BoundedStack | UnboundedStack
	rtos: str
	membership: Membership = Membership.NAMES
	never_linked_entry_points: int | None = None
	"""Counted apart from ``discarded_entry_points`` only when the linker decides membership."""
	phantom_libcalls: tuple[str, ...] | None = None
	"""Libcalls a kept function's ``.ci`` records but the final link did not keep; only when
	the linker decides membership."""
