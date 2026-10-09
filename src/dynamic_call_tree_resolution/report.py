# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""JSON-serializable analysis reports."""

from itertools import chain, groupby
from typing import TYPE_CHECKING, Final, assert_never

from msgspec import Struct
from salix import Struct as SalixStruct

from dynamic_call_tree_resolution.call_sites import (
	Dispatched,
	Null,
	call_site_candidates,
	calls_nothing,
	narrowing_signature,
	resolve,
	signature_narrowings,
	site_targets,
	target_addresses,
)
from dynamic_call_tree_resolution.callgraph import EdgeKind
from dynamic_call_tree_resolution.field_narrowing import narrowed, narrowing_at
from dynamic_call_tree_resolution.linker import Membership
from dynamic_call_tree_resolution.model import (
	BARE_METAL,
	Address,
	Dead,
	FunctionSignature,
	Machine,
	NullSlot,
	Provenance,
	Residue,
	SkipReason,
	UnresolvedSlot,
	aligned,
	render_path,
	thumb_twin,
)
from dynamic_call_tree_resolution.pexplorer import (
	DynamicSites,
	PexplorerReport,
	dynamic_sites_by_caller,
)
from dynamic_call_tree_resolution.points_to import (
	not_enumerated,
	null_slots,
	signatures_by_slot,
	unresolved_slots,
)
from dynamic_call_tree_resolution.stack_analysis import (
	Bounded,
	LevelSource,
	Nested,
	Nesting,
	Reason,
	Unbounded,
)
from dynamic_call_tree_resolution.vsa import address_taken, referrers
from dynamic_call_tree_resolution.vsa.links import function_covering

if TYPE_CHECKING:
	from collections.abc import Mapping

	from dynamic_call_tree_resolution.field_narrowing import NarrowedSpan
	from dynamic_call_tree_resolution.model import CallSite, Program, RtosModel, SlotAssignment


class Candidate(Struct):
	"""A resolved target function."""

	name: str
	address: int


class SlotAssignmentReport(Struct):
	"""One resolved function-pointer slot, rendered for consumers."""

	slot_address: int
	member_path: str
	candidates: tuple[Candidate, ...]
	provenance: Provenance
	relocated: bool


class SignatureReport(Struct):
	"""A function signature rendered for consumers."""

	return_type: str
	parameters: tuple[str, ...]


class CallSiteReport(Struct):
	"""One extracted indirect call site, rendered for consumers."""

	caller: str
	site_address: int
	slot_address: int | None
	member_path: str | None
	candidates: tuple[Candidate, ...]
	field: str | None
	"""The struct field ``--narrow-by-field`` narrowed the candidates to; unsound under casts."""
	signature: SignatureReport | None = None
	"""The slot signature ``--narrow-by-signature`` narrowed the candidates to; unsound under
	casts."""
	external: str | None = None
	"""The undefined symbol whose address the dynamic loader writes to the slot."""
	null: bool = False
	"""The target can only be address 0, so the site calls nothing."""
	dispatch: bool = False
	"""The candidates are the entries the RTOS model says the site's dispatch loop calls."""
	dead: bool = False
	"""Only branches the analysis's values decide against lead to the site, so it calls
	nothing."""


class UnresolvedSlotReport(Struct):
	"""A function-pointer slot with no statically resolved candidates."""

	slot_address: int
	member_path: str
	signature: SignatureReport | None
	residue: Residue


class NullSlotReport(Struct):
	"""A read-only function-pointer slot that holds NULL: resolved, to nothing."""

	slot_address: int
	member_path: str
	signature: SignatureReport | None


class NotEnumeratedReport(Struct):
	"""An object or member whose slots the analysis does not enumerate."""

	member_path: str
	reason: SkipReason


class ThreadReport(Struct):
	"""One thread of the RTOS model."""

	name: str
	entry: Candidate
	arguments: tuple[int, ...]
	seeded: bool
	"""The entry started from these arguments; otherwise from unknown ones."""


class RtosReport(Struct):
	"""The RTOS model an analysis used."""

	name: str
	evidence: tuple[str, ...]
	threads: tuple[ThreadReport, ...]


BARE_METAL_REPORT: Final = RtosReport(name=BARE_METAL.name, evidence=(), threads=())


class AnalysisReport(Struct):
	"""The analysis report of one program."""

	assignments: tuple[SlotAssignmentReport, ...]
	call_sites: tuple[CallSiteReport, ...]
	unresolved_slots: tuple[UnresolvedSlotReport, ...]
	null_slots: tuple[NullSlotReport, ...]
	not_enumerated: tuple[NotEnumeratedReport, ...]
	total_slots: int
	resolved_slots: int
	resolved_targets: int
	rtos: RtosReport


class FunctionComparison(Struct):
	"""Per-function dynamic-call comparison between pexplorer and dctr."""

	address: int
	caller: str
	pexplorer_dynamic_sites: int
	dctr_call_sites: int
	dctr_resolved_sites: int
	dctr_exact_sites: int


class ComparisonReport(Struct):
	"""Resolution rollup of one ELF, for cross-tool comparison."""

	elf: str
	machine: Machine
	functions: int
	total_slots: int
	resolved_slots: int
	unresolved_slots: int
	call_sites: int
	resolved_call_sites: int
	exact_call_sites: int
	candidate_size_counts: tuple[tuple[int, int], ...]
	pexplorer_dynamic_sites: int | None = None
	function_comparisons: tuple[FunctionComparison, ...] = ()
	rtos: str = BARE_METAL.name


class BoundedStack(Struct, tag="bounded", tag_field="kind"):
	"""A stack depth that bounds every path of the call graph."""

	bytes: int
	measured: tuple[str, ...] = ()
	"""Reachable functions whose frames come from their code."""
	stack_reservation_bytes: int = 0
	"""The top of a thread's stack the RTOS model takes before the thread runs, included in
	``bytes``."""
	exception_frame_bytes: int = 0
	"""The interrupt frame the RTOS model adds to a thread's depth, included in ``bytes``."""
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
	"""The interrupt frame the RTOS model adds to a thread's depth, included in
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


def stack_bound_report(bound: Bounded | Unbounded) -> BoundedStack | UnboundedStack:
	match bound:
		case Bounded(bytes=depth, measured=measured, exception_frame=exception_frame):
			return BoundedStack(
				bytes=depth,
				measured=tuple(sorted(measured)),
				stack_reservation_bytes=bound.stack_reservation,
				exception_frame_bytes=exception_frame,
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
				assumed_no_recursion=tuple(sorted(bound.assumed_no_recursion)),
				narrowed_by_field=tuple(sorted(bound.narrowed_by_field)),
				narrowed_by_signature=tuple(sorted(bound.narrowed_by_signature)),
				assumed_unwritten=tuple(sorted(bound.assumed_unwritten)),
				assumed_frames=tuple(sorted(bound.assumed_frames)),
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


class ReferrerReport(Struct):
	"""One place that holds or computes an address-taken function's address."""

	slot: int
	holder: str | None
	"""The function or data object spanning the slot; none outside every symbol."""


class AddressTakenReport(Struct):
	"""One row of ``referrers --json``."""

	name: str
	address: int
	referrers: tuple[ReferrerReport, ...]


def referrers_report(program: Program) -> tuple[AddressTakenReport, ...]:
	return tuple(
		sorted(
			(
				AddressTakenReport(
					name=program.functions[function].name,
					address=function,
					referrers=tuple(
						ReferrerReport(slot=slot, holder=_holder(program, slot))
						for slot in sorted(slots)
					),
				)
				for function, slots in referrers(program, address_taken(program)).items()
			),
			key=_name_and_address,
		)
	)


def _name_and_address(report: AddressTakenReport) -> tuple[str, int]:
	return (report.name, report.address)


def _holder(program: Program, slot: Address) -> str | None:
	return next(
		chain(
			(
				function.name
				for function in (function_covering(program, slot),)
				if function is not None
			),
			(
				data_object.name
				for data_object in program.objects.values()
				if slot - data_object.address in range(max(data_object.size, 1))
			),
		),
		None,
	)


class SlotCounts(SalixStruct):
	"""The counts every report builder shares."""

	resolved_slots: int
	total_slots: int
	resolved_targets: int


def slot_counts(
	resolved: tuple[SlotAssignment, ...],
	unresolved: tuple[UnresolvedSlot, ...],
	nulls: tuple[NullSlot, ...],
) -> SlotCounts:
	slots = frozenset(assignment.slot for assignment in resolved) | {slot.slot for slot in nulls}
	return SlotCounts(
		resolved_slots=len(slots),
		total_slots=len(slots | {slot.slot for slot in unresolved}),
		resolved_targets=sum(len(assignment.candidates) for assignment in resolved),
	)


def resolved_by_slot(resolved: tuple[SlotAssignment, ...]) -> Mapping[Address, SlotAssignment]:
	return {assignment.slot: assignment for assignment in resolved}


def _site_address(site: CallSite) -> Address:
	return aligned(site.caller_address)


def _candidates(program: Program, addresses: frozenset[Address]) -> tuple[Candidate, ...]:
	return tuple(
		Candidate(name=program.functions[address].name, address=address)
		for address in sorted(addresses)
	)


def rtos_report(program: Program, rtos: RtosModel, seeded: frozenset[str]) -> RtosReport:
	return RtosReport(
		name=rtos.name,
		evidence=rtos.evidence,
		threads=tuple(
			ThreadReport(
				name=thread.name,
				entry=Candidate(name=program.functions[thread.entry].name, address=thread.entry),
				arguments=thread.arguments,
				seeded=thread.name in seeded,
			)
			for thread in rtos.threads
		),
	)


def build_report(
	program: Program,
	resolved: tuple[SlotAssignment, ...],
	call_sites: tuple[CallSite, ...],
	*,
	narrow_by_signature: bool = False,
	narrowed_by_field: tuple[NarrowedSpan, ...] = (),
	rtos: RtosReport = BARE_METAL_REPORT,
	fallback: frozenset[Address] | None = None,
) -> AnalysisReport:
	unresolved = unresolved_slots(program, resolved)
	nulls = null_slots(program, resolved)
	resolved_map = resolved_by_slot(resolved)
	slot_paths = {
		**{slot.slot: slot.path for slot in unresolved},
		**{slot.slot: slot.path for slot in nulls},
		**{slot: assignment.path for slot, assignment in resolved_map.items()},
	}
	narrowings = (
		signature_narrowings(program, signatures_by_slot(unresolved), fallback)
		if narrow_by_signature
		else None
	)
	counts = slot_counts(resolved, unresolved, nulls)
	return AnalysisReport(
		assignments=tuple(
			SlotAssignmentReport(
				slot_address=assignment.slot,
				member_path=render_path(assignment.path),
				candidates=_candidates(program, assignment.candidates),
				provenance=assignment.provenance,
				relocated=assignment.relocated,
			)
			for assignment in resolved
		),
		call_sites=tuple(
			CallSiteReport(
				caller=program.functions[site.caller_address].name,
				site_address=site.site_address,
				slot_address=site.slot,
				member_path=next(
					(
						render_path(slot_paths[slot])
						for slot in (site.loaded_from, site.slot)
						if slot is not None and slot in slot_paths
					),
					None,
				),
				candidates=_candidates(
					program, narrowing.targets if narrowing is not None else chased
				),
				field=(
					f"{narrowing.field.record}.{narrowing.field.member}"
					if narrowing is not None
					else None
				),
				signature=_render_signature(narrowing_signature(findings)),
				external=site.external,
				null=any(isinstance(found, Null) for found in findings),
				dispatch=any(isinstance(found, Dispatched) for found in findings),
				dead=any(isinstance(found, Dead) for found in findings),
			)
			for site in call_sites
			for findings in (site_targets(program, site, resolved_map, narrowings),)
			for chased in (
				frozenset(target for found in findings for target in target_addresses(found)),
			)
			for narrowing in (
				None
				if chased or calls_nothing(program, site)
				else narrowing_at(narrowed_by_field, site.site_address),
			)
		),
		unresolved_slots=tuple(
			UnresolvedSlotReport(
				slot_address=slot.slot,
				member_path=render_path(slot.path),
				signature=_render_signature(slot.signature),
				residue=slot.residue,
			)
			for slot in unresolved
		),
		null_slots=tuple(
			NullSlotReport(
				slot_address=slot.slot,
				member_path=render_path(slot.path),
				signature=_render_signature(slot.signature),
			)
			for slot in nulls
		),
		not_enumerated=tuple(
			NotEnumeratedReport(member_path=render_path(item.path), reason=item.reason)
			for item in not_enumerated(program)
		),
		total_slots=counts.total_slots,
		resolved_slots=counts.resolved_slots,
		resolved_targets=counts.resolved_targets,
		rtos=rtos,
	)


def _render_signature(signature: FunctionSignature | None) -> SignatureReport | None:
	return (
		SignatureReport(return_type=signature.return_type, parameters=signature.parameters)
		if signature is not None
		else None
	)


def build_comparison(
	name: str,
	program: Program,
	pexplorer: PexplorerReport | None = None,
	*,
	narrow_by_signature: bool = False,
	narrowed_by_field: tuple[NarrowedSpan, ...] = (),
	rtos: RtosModel = BARE_METAL,
) -> ComparisonReport:
	resolution = resolve(program, rtos)
	resolved = resolution.assignments
	unresolved = unresolved_slots(program, resolved)
	resolved_map = resolved_by_slot(resolved)
	narrowings = (
		signature_narrowings(program, signatures_by_slot(unresolved), resolution.fallback)
		if narrow_by_signature
		else None
	)
	counts = slot_counts(resolved, unresolved, null_slots(program, resolved))
	outcomes = tuple(
		_SiteOutcome(
			caller=_site_address(site),
			candidates=0
			if nothing
			else len(
				narrowed(
					call_site_candidates(program, site, resolved_map, narrowings),
					narrowed_by_field,
					site.site_address,
				)
			),
			calls_nothing=nothing,
		)
		for site in resolution.sites
		for nothing in (calls_nothing(program, site),)
	)
	candidate_sizes = sorted(outcome.candidates for outcome in outcomes)
	dynamic_by_caller: Mapping[Address, DynamicSites] = (
		dynamic_sites_by_caller(pexplorer) if pexplorer is not None else {}
	)
	sites_by_caller = {
		caller_address: tuple(group)
		for caller_address, group in groupby(
			sorted(outcomes, key=_outcome_caller), key=_outcome_caller
		)
	}
	rows = (
		*(
			FunctionComparison(
				address=caller_address,
				caller=_caller_name(program, caller_address),
				pexplorer_dynamic_sites=dynamic_by_caller[caller_address].total
				if caller_address in dynamic_by_caller
				else 0,
				dctr_call_sites=len(group),
				dctr_resolved_sites=sum(map(_resolved, group)),
				dctr_exact_sites=sum(map(_exact, group)),
			)
			for caller_address, group in sites_by_caller.items()
		),
		*(
			FunctionComparison(
				address=caller_address,
				caller=", ".join(dynamic_by_caller[caller_address].names),
				pexplorer_dynamic_sites=dynamic_by_caller[caller_address].total,
				dctr_call_sites=0,
				dctr_resolved_sites=0,
				dctr_exact_sites=0,
			)
			for caller_address in dynamic_by_caller
			if caller_address not in sites_by_caller
		),
	)
	return ComparisonReport(
		elf=name,
		machine=program.machine,
		functions=len(program.functions),
		total_slots=counts.total_slots,
		resolved_slots=counts.resolved_slots,
		unresolved_slots=len(unresolved),
		call_sites=len(candidate_sizes),
		resolved_call_sites=sum(map(_resolved, outcomes)),
		exact_call_sites=sum(map(_exact, outcomes)),
		candidate_size_counts=tuple(
			(size, candidate_sizes.count(size)) for size in dict.fromkeys(candidate_sizes, 0)
		),
		pexplorer_dynamic_sites=(
			sum(sites.total for sites in dynamic_by_caller.values())
			if pexplorer is not None
			else None
		),
		function_comparisons=tuple(sorted(rows, key=_row_caller)),
		rtos=rtos.name,
	)


def _row_caller(row: FunctionComparison) -> str:
	return row.caller


class _SiteOutcome(Struct):
	"""What a comparison counts of one call site."""

	caller: Address
	candidates: int
	calls_nothing: bool


def _outcome_caller(outcome: _SiteOutcome) -> Address:
	return outcome.caller


def _resolved(outcome: _SiteOutcome) -> bool:
	return outcome.candidates > 0 or outcome.calls_nothing


def _exact(outcome: _SiteOutcome) -> bool:
	return outcome.candidates == 1 or outcome.calls_nothing


def _caller_name(program: Program, caller_address: Address) -> str:
	function = program.functions.get(caller_address) or program.functions.get(
		thumb_twin(caller_address)
	)
	if function is None:
		raise ValueError(f"no function for caller address {caller_address:#x}")  # pragma: no cover
	return function.name
