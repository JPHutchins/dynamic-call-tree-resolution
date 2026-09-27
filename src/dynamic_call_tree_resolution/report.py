# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""JSON-serializable analysis reports."""

from typing import TYPE_CHECKING, NamedTuple

from msgspec import Struct

from dynamic_call_tree_resolution.call_sites import call_site_candidates, extract_call_sites
from dynamic_call_tree_resolution.model import (
	ANONYMOUS,
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
from dynamic_call_tree_resolution.points_to import assignments, signatures_by_slot, unresolved_slots

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


class CallSiteReport(Struct):
	"""One extracted indirect call site with its resolved candidates."""

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
	"""All resolved slots, call sites, and unresolved slots of one program."""

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


class AnalysisSummary(Struct):
	"""Cross-cutting resolution and stack summary for CI reporting."""

	resolved_slots: int
	total_slots: int
	unresolved_slots: int
	resolved_targets: int
	indirect_call_sites: int
	total_functions: int
	entry_points: int
	worst_case_bytes: int
	worst_case_entry: str


class SlotCounts(NamedTuple):
	"""Slot and target counts shared by every report builder."""

	resolved_slots: int
	total_slots: int
	resolved_targets: int


def slot_counts(
	resolved: tuple[SlotAssignment, ...], unresolved: tuple[UnresolvedSlot, ...]
) -> SlotCounts:
	"""The resolution rollup shared by the report, comparison, and summary."""
	slots = frozenset(assignment.slot for assignment in resolved)
	return SlotCounts(
		resolved_slots=len(slots),
		total_slots=len(slots | {slot.slot for slot in unresolved}),
		resolved_targets=sum(len(assignment.candidates) for assignment in resolved),
	)


def resolved_by_slot(resolved: tuple[SlotAssignment, ...]) -> Mapping[Address, SlotAssignment]:
	"""Resolved assignments keyed by slot address."""
	return {assignment.slot: assignment for assignment in resolved}


def _candidates(program: Program, addresses: frozenset[Address]) -> tuple[Candidate, ...]:
	"""Candidate records for the given target addresses."""
	return tuple(
		Candidate(name=program.functions[address].name, address=address)
		for address in sorted(addresses)
	)


def build_report(
	program: Program,
	resolved: tuple[SlotAssignment, ...],
	call_sites: tuple[CallSite, ...],
) -> AnalysisReport:
	"""Render resolved assignments, per-site resolutions, and unresolved slots."""
	unresolved = unresolved_slots(program, resolved)
	resolved_map = resolved_by_slot(resolved)
	signatures = signatures_by_slot(unresolved)
	counts = slot_counts(resolved, unresolved)
	return AnalysisReport(
		assignments=tuple(
			SlotAssignmentReport(
				slot_address=assignment.slot,
				member_path=render_path(assignment.path),
				candidates=_candidates(program, assignment.candidates),
				provenance=assignment.provenance,
			)
			for assignment in resolved
		),
		call_sites=tuple(
			CallSiteReport(
				caller=program.functions[site.caller_address].name,
				site_address=site.site_address,
				slot_address=site.slot,
				member_path=(
					render_path(resolved_map[site.slot].path)
					if site.slot is not None and site.slot in resolved_map
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
	name: str, program: Program, pexplorer: PexplorerReport | None = None
) -> ComparisonReport:
	"""Resolution rollup of one ELF, for cross-tool comparison.

	When a pexplorer report is given, each function with dynamic calls on
	either side gets a row: pexplorer's dynamic-call count against dctr's
	per-site candidate sets.
	"""
	resolved = assignments(program)
	unresolved = unresolved_slots(program, resolved)
	resolved_map = resolved_by_slot(resolved)
	signatures = signatures_by_slot(unresolved)
	counts = slot_counts(resolved, unresolved)
	sites = extract_call_sites(program)
	candidate_sizes = sorted(
		len(call_site_candidates(program, site, resolved_map, signatures)) for site in sites
	)
	dynamic_by_caller: Mapping[Address, DynamicSites] = (
		dynamic_sites_by_caller(pexplorer) if pexplorer is not None else {}
	)
	sites_by_caller: dict[Address, list[frozenset[Address]]] = {}
	for site in sites:
		sites_by_caller.setdefault(aligned(site.caller_address), []).append(
			call_site_candidates(program, site, resolved_map, signatures)
		)
	rows = [
		FunctionComparison(
			address=caller_address,
			caller=_row_name(program, caller_address, dynamic_by_caller),
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
			(size, candidate_sizes.count(size)) for size in dict.fromkeys(candidate_sizes)
		),
		pexplorer_dynamic_sites=(
			sum(sites.total for sites in dynamic_by_caller.values())
			if pexplorer is not None
			else None
		),
		function_comparisons=tuple(sorted(rows, key=lambda row: row.caller)),
	)


def _caller_name(program: Program, caller_address: Address) -> str:
	function = program.functions.get(caller_address) or program.functions.get(
		thumb_twin(caller_address)
	)
	if function is None:
		raise ValueError(f"no function for caller address {caller_address:#x}")  # pragma: no cover
	return function.name


def _row_name(
	program: Program,
	caller_address: Address,
	dynamic_by_caller: Mapping[Address, DynamicSites],
) -> str:
	name = _caller_name(program, caller_address)
	if name == ANONYMOUS and caller_address in dynamic_by_caller:
		return ", ".join(dynamic_by_caller[caller_address].names)
	return name
