# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""What the stack analysis works over and what it reports."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from salix import Struct

from dynamic_call_tree_resolution.stack_usage import StackUsage

if TYPE_CHECKING:
	from collections.abc import Mapping

	from dynamic_call_tree_resolution.callgraph import CallPair, EdgeKind


class Bounded(Struct):
	"""A depth that bounds every path of the call graph as given."""

	bytes: int
	measured: frozenset[str] = frozenset()
	"""Reachable functions whose frames come from their code."""
	stack_reservation: int = 0
	"""The top of a thread's stack the RTOS model takes before the thread runs, included in
	``bytes``."""
	exception_frame: int = 0
	"""The interrupt frame the RTOS model adds to a thread's depth, included in ``bytes``."""
	assumed_no_recursion: frozenset[str] = frozenset()
	"""Reachable functions assumed never to call themselves."""
	narrowed_by_field: frozenset[str] = frozenset()
	"""Reachable callers whose indirect call ``--narrow-by-field`` narrowed."""
	narrowed_by_signature: frozenset[str] = frozenset()
	"""Reachable callers whose indirect call ``--narrow-by-signature`` narrowed."""
	assumed_unwritten: frozenset[str] = frozenset()
	"""Reachable callers whose indirect calls rest on ``--assume-unwritten``."""
	assumed_frames: frozenset[str] = frozenset()
	"""Reachable functions whose frames ``--assume-frame`` states."""


class Unbounded(Struct):
	"""A depth that bounds only from below."""

	at_least: int
	recursion: frozenset[str]
	"""Reachable functions on a cycle."""
	unmeasured: frozenset[str]
	"""Reachable functions with neither a ``.su`` record nor code that can be measured."""
	dynamic: frozenset[str]
	"""Reachable frames GCC could not bound."""
	unresolved: frozenset[str]
	"""Reachable callers of an indirect call without candidates."""
	measured: frozenset[str] = frozenset()
	"""Reachable functions whose frames come from their code."""
	stack_reservation: int = 0
	"""The top of a thread's stack the RTOS model takes before the thread runs, included in
	``at_least``."""
	exception_frame: int = 0
	"""The interrupt frame the RTOS model adds to a thread's depth, included in ``at_least``."""
	assumed_no_recursion: frozenset[str] = frozenset()
	"""Reachable functions assumed never to call themselves."""
	narrowed_by_field: frozenset[str] = frozenset()
	"""Reachable callers whose indirect call ``--narrow-by-field`` narrowed."""
	narrowed_by_signature: frozenset[str] = frozenset()
	"""Reachable callers whose indirect call ``--narrow-by-signature`` narrowed."""
	assumed_unwritten: frozenset[str] = frozenset()
	"""Reachable callers whose indirect calls rest on ``--assume-unwritten``."""
	assumed_frames: frozenset[str] = frozenset()
	"""Reachable functions whose frames ``--assume-frame`` states."""


class LevelSource(StrEnum):
	"""Where the most configurable-priority exceptions one nesting chain holds comes from."""

	DEVICETREE = "devicetree"
	VECTOR_TABLE = "vector table"


class PriorityLevels(Struct):
	"""The most configurable-priority exceptions one nesting chain holds."""

	count: int
	source: LevelSource


class Nested(Struct):
	"""One exception an interrupt stack's depth stacks."""

	exception: int
	handler: str
	depth: int
	"""The handler's depth, or its lower bound."""


class Nesting(Struct):
	"""How an interrupt stack's depth stacks exceptions on the code that starts on it."""

	base: Nested
	chain: tuple[Nested, ...]
	"""The nested exceptions' handlers, in the order the depth adds them."""
	levels: PriorityLevels
	exception_frame: int
	"""What each nested exception's hardware frame adds."""


class StackReport(Struct):
	"""One worst-case stack depth."""

	entry: str
	bound: Bounded | Unbounded
	nesting: Nesting | None = None
	stack_size: int | None = None
	"""The size of the stack the entry runs on, as the RTOS model declares it."""


class Handled(Struct):
	"""One exception's handler, with the handler's depth."""

	exception: int
	report: StackReport


class Reason(StrEnum):
	"""What keeps a depth from bounding every path, or, measured, what the depth rests on."""

	RECURSION = "recursion"
	UNMEASURED = "unmeasured"
	DYNAMIC = "dynamic"
	UNRESOLVED = "unresolved"
	MEASURED = "measured"
	ASSUMED_NO_RECURSION = "assumed no recursion"
	NARROWED_BY_FIELD = "narrowed by field"
	NARROWED_BY_SIGNATURE = "narrowed by signature"
	ASSUMED_UNWRITTEN = "assumed unwritten"
	ASSUMED_FRAME = "assumed frame"


class FunctionReason(Struct):
	"""A reason a depth carries, and the function it comes from."""

	reason: Reason
	function: str


type Reasons = frozenset[FunctionReason]


class MeasuredFrame(Struct):
	"""A frame measured from the code of a function GCC left no ``.su`` record for."""

	function: str
	bytes: int


class OwnFrame(Struct):
	"""A function's own frame, measured from its code; its callees count through its edges."""

	function: str
	bytes: int


class AssumedFrame(Struct):
	"""A frame ``--assume-frame`` states for a function nothing measures."""

	function: str
	bytes: int


type Frame = StackUsage | MeasuredFrame | OwnFrame | AssumedFrame


class StackGraph(Struct):
	"""A call graph with the frames and depths its stack bounds come from."""

	adjacency: Mapping[str, frozenset[str]]
	frame_by_name: Mapping[str, Frame]
	components: Mapping[str, frozenset[str]]
	"""Each node's strongly connected component."""
	depths: Mapping[str, int]
	reasons: Mapping[str, Reasons]
	kinds: Mapping[CallPair, frozenset[EdgeKind]]
	roots: tuple[str, ...]
	assumed_no_recursion: frozenset[str] = frozenset()
	"""The functions whose call to themselves an assumption removed."""
	narrowings: Mapping[str, frozenset[Reason]] = {}
	"""The narrowings each node's own indirect calls rest on."""


class PathStep(Struct):
	"""One function on a deepest path."""

	function: str
	frame: int
	cumulative: int
	edge: frozenset[EdgeKind]
	"""How the previous step calls this one."""
	flags: frozenset[Reason]
	"""What this function's own frame, calls and cycle add to the depth's reasons."""


MAX_CYCLE_SIZE: Final = 16


INDIRECT_CALLEE: Final = "__indirect_call"


NO_CALLEE: Final = "__no_call"


def frame_key(name: str) -> str:
	"""Reduce a VCG or ``.su`` symbol to its bare assembly form.

	>>> frame_key("/abs/path/file.c:func")
	'func'
	>>> frame_key("func.isra.0")
	'func'
	>>> frame_key("func.constprop.0.isra.0"), frame_key("func.constprop.isra")
	('func', 'func')
	>>> frame_key("func.localalias"), frame_key("func.cold"), frame_key("func.lto_priv.0")
	('func', 'func', 'func')
	>>> frame_key("func.part_of"), frame_key("func.isra.0.x")
	('func.part_of', 'func.isra.0.x')
	"""
	return re.sub(
		r"(?:\.(?:isra|constprop|part|localalias|cold|lto_priv)(?:\.\d+)?)+$",
		"",
		name.rsplit(":", 1)[-1] if "/" in name else name,
	)
