# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""JSON-serializable analysis reports."""

from itertools import groupby
from typing import TYPE_CHECKING, assert_never

from msgspec import Struct
from salix import Struct as SalixStruct

from dynamic_call_tree_resolution.call_sites import call_site_candidates, resolve
from dynamic_call_tree_resolution.model import (
	Address,
	FunctionSignature,
	Machine,
	Provenance,
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
from dynamic_call_tree_resolution.points_to import signatures_by_slot, unresolved_slots
from dynamic_call_tree_resolution.stack_analysis import Bounded, Unbounded

if TYPE_CHECKING:
	from collections.abc import Mapping

	from dynamic_call_tree_resolution.model import CallSite, Program, SlotAssignment


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


class CallSiteReport(Struct):
	"""One extracted indirect call site, rendered for consumers."""

	caller: str
	site_address: int
	slot_address: int | None
	member_path: str | None
	candidates: tuple[Candidate, ...]


class SignatureReport(Struct):
	"""A function signature rendered for consumers."""

	return_type: str
	parameters: tuple[str, ...]


class UnresolvedSlotReport(Struct):
	"""A function-pointer slot with no statically resolved candidates."""

	slot_address: int
	member_path: str
	signature: SignatureReport | None


class AnalysisReport(Struct):
	"""The analysis report of one program."""

	assignments: tuple[SlotAssignmentReport, ...]
	call_sites: tuple[CallSiteReport, ...]
	unresolved_slots: tuple[UnresolvedSlotReport, ...]
	total_slots: int
	resolved_slots: int
	resolved_targets: int


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


class BoundedStack(Struct, tag="bounded", tag_field="kind"):
	"""A stack depth that bounds every path of the call graph."""

	bytes: int


class UnboundedStack(Struct, tag="unbounded", tag_field="kind"):
	"""The JSON form of ``Unbounded``."""

	at_least_bytes: int
	recursion: tuple[str, ...]
	unmeasured: tuple[str, ...]
	dynamic: tuple[str, ...]
	unresolved: tuple[str, ...]


def stack_bound_report(bound: Bounded | Unbounded) -> BoundedStack | UnboundedStack:
	match bound:
		case Bounded(bytes=depth):
			return BoundedStack(bytes=depth)
		case Unbounded():
			return UnboundedStack(
				at_least_bytes=bound.at_least,
				recursion=tuple(sorted(bound.recursion)),
				unmeasured=tuple(sorted(bound.unmeasured)),
				dynamic=tuple(sorted(bound.dynamic)),
				unresolved=tuple(sorted(bound.unresolved)),
			)
		case _ as unreachable:
			assert_never(unreachable)


class StackEntryReport(Struct):
	"""One row of ``stack --json``."""

	entry: str
	bound: BoundedStack | UnboundedStack


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


class SlotCounts(SalixStruct):
	"""The counts every report builder shares."""

	resolved_slots: int
	total_slots: int
	resolved_targets: int


def slot_counts(
	resolved: tuple[SlotAssignment, ...], unresolved: tuple[UnresolvedSlot, ...]
) -> SlotCounts:
	slots = frozenset(assignment.slot for assignment in resolved)
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


def build_report(
	program: Program,
	resolved: tuple[SlotAssignment, ...],
	call_sites: tuple[CallSite, ...],
	*,
	narrow_by_signature: bool = False,
) -> AnalysisReport:
	unresolved = unresolved_slots(program, resolved)
	resolved_map = resolved_by_slot(resolved)
	slot_paths = {
		**{slot.slot: slot.path for slot in unresolved},
		**{slot: assignment.path for slot, assignment in resolved_map.items()},
	}
	signatures = signatures_by_slot(unresolved) if narrow_by_signature else None
	counts = slot_counts(resolved, unresolved)
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
				member_path=(
					render_path(slot_paths[site.slot])
					if site.slot is not None and site.slot in slot_paths
					else None
				),
				candidates=_candidates(
					program, call_site_candidates(program, site, resolved_map, signatures)
				),
			)
			for site in call_sites
		),
		unresolved_slots=tuple(
			UnresolvedSlotReport(
				slot_address=slot.slot,
				member_path=render_path(slot.path),
				signature=_render_signature(slot.signature),
			)
			for slot in unresolved
		),
		total_slots=counts.total_slots,
		resolved_slots=counts.resolved_slots,
		resolved_targets=counts.resolved_targets,
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
) -> ComparisonReport:
	resolution = resolve(program)
	resolved = resolution.assignments
	unresolved = unresolved_slots(program, resolved)
	resolved_map = resolved_by_slot(resolved)
	signatures = signatures_by_slot(unresolved) if narrow_by_signature else None
	counts = slot_counts(resolved, unresolved)
	sites = resolution.sites
	candidate_sizes = sorted(
		len(call_site_candidates(program, site, resolved_map, signatures)) for site in sites
	)
	dynamic_by_caller: Mapping[Address, DynamicSites] = (
		dynamic_sites_by_caller(pexplorer) if pexplorer is not None else {}
	)
	sites_by_caller = {
		caller_address: [
			call_site_candidates(program, site, resolved_map, signatures) for site in group
		]
		for caller_address, group in groupby(sorted(sites, key=_site_address), key=_site_address)
	}
	rows = [
		FunctionComparison(
			address=caller_address,
			caller=_caller_name(program, caller_address),
			pexplorer_dynamic_sites=dynamic_by_caller[caller_address].total
			if caller_address in dynamic_by_caller
			else 0,
			dctr_call_sites=len(candidates),
			dctr_resolved_sites=sum(bool(candidate_set) for candidate_set in candidates),
			dctr_exact_sites=sum(len(candidate_set) == 1 for candidate_set in candidates),
		)
		for caller_address, candidates in sites_by_caller.items()
	]
	rows.extend(
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
	)
	return ComparisonReport(
		elf=name,
		machine=program.machine,
		functions=len(program.functions),
		total_slots=counts.total_slots,
		resolved_slots=counts.resolved_slots,
		unresolved_slots=len(unresolved),
		call_sites=len(candidate_sizes),
		resolved_call_sites=sum(size > 0 for size in candidate_sizes),
		exact_call_sites=sum(size == 1 for size in candidate_sizes),
		candidate_size_counts=tuple(
			(size, candidate_sizes.count(size)) for size in dict.fromkeys(candidate_sizes, 0)
		),
		pexplorer_dynamic_sites=(
			sum(sites.total for sites in dynamic_by_caller.values())
			if pexplorer is not None
			else None
		),
		function_comparisons=tuple(sorted(rows, key=_row_caller)),
	)


def _row_caller(row: FunctionComparison) -> str:
	return row.caller


def _caller_name(program: Program, caller_address: Address) -> str:
	function = program.functions.get(caller_address) or program.functions.get(
		thumb_twin(caller_address)
	)
	if function is None:
		raise ValueError(f"no function for caller address {caller_address:#x}")  # pragma: no cover
	return function.name
