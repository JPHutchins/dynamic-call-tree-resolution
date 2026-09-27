# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""JSON-serializable analysis reports."""

from typing import TYPE_CHECKING

from msgspec import Struct

from dynamic_call_tree_resolution.call_sites import call_site_candidates, extract_call_sites
from dynamic_call_tree_resolution.model import Address, FunctionSignature, Provenance, render_path
from dynamic_call_tree_resolution.pexplorer import PexplorerReport, dynamic_sites_by_caller
from dynamic_call_tree_resolution.points_to import assignments, unresolved_slots

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
	machine: str
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


def build_report(
	program: Program,
	resolved: tuple[SlotAssignment, ...],
	call_sites: tuple[CallSite, ...],
) -> AnalysisReport:
	"""Render resolved assignments, per-site resolutions, and unresolved slots."""
	resolved_by_slot = {assignment.slot: assignment for assignment in resolved}
	unresolved = unresolved_slots(program, resolved)
	return AnalysisReport(
		assignments=tuple(
			SlotAssignmentReport(
				slot_address=assignment.slot,
				member_path=render_path(assignment.path),
				candidates=tuple(
					Candidate(name=program.functions[address].name, address=address)
					for address in sorted(assignment.candidates)
				),
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
					render_path(resolved_by_slot[site.slot].path)
					if site.slot is not None and site.slot in resolved_by_slot
					else None
				),
				candidates=tuple(
					Candidate(name=program.functions[address].name, address=address)
					for address in sorted(call_site_candidates(program, site, resolved_by_slot))
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
		total_slots=len(set(resolved_by_slot) | {slot.slot for slot in unresolved}),
		resolved_slots=len(resolved_by_slot),
		resolved_targets=sum(len(assignment.candidates) for assignment in resolved),
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
	resolved_by_slot = {assignment.slot: assignment for assignment in resolved}
	sites = extract_call_sites(program)
	candidate_sizes = sorted(
		len(call_site_candidates(program, site, resolved_by_slot)) for site in sites
	)
	dynamic_by_caller: Mapping[Address, tuple[tuple[str, ...], int]] = (
		dynamic_sites_by_caller(pexplorer) if pexplorer is not None else {}
	)
	sites_by_caller: dict[Address, list[frozenset[Address]]] = {}
	for site in sites:
		sites_by_caller.setdefault(Address(site.caller_address & ~1), []).append(
			call_site_candidates(program, site, resolved_by_slot)
		)
	rows = [
		FunctionComparison(
			address=caller_address,
			caller=_row_name(program, caller_address, dynamic_by_caller),
			pexplorer_dynamic_sites=dynamic_by_caller[caller_address][1]
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
			caller=", ".join(dynamic_by_caller[caller_address][0]),
			pexplorer_dynamic_sites=dynamic_by_caller[caller_address][1],
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
		total_slots=len(set(resolved_by_slot) | {slot.slot for slot in unresolved}),
		resolved_slots=len(resolved_by_slot),
		unresolved_slots=len(unresolved),
		call_sites=len(candidate_sizes),
		resolved_call_sites=sum(size > 0 for size in candidate_sizes),
		exact_call_sites=sum(size == 1 for size in candidate_sizes),
		candidate_size_counts=tuple(
			(size, candidate_sizes.count(size)) for size in dict.fromkeys(candidate_sizes)
		),
		pexplorer_dynamic_sites=(
			sum(count for _, count in dynamic_by_caller.values()) if pexplorer is not None else None
		),
		function_comparisons=tuple(sorted(rows, key=lambda row: row.caller)),
	)


def _caller_name(program: Program, caller_address: Address) -> str:
	function = program.functions.get(caller_address) or program.functions.get(
		Address(caller_address | 1)
	)
	if function is None:
		raise ValueError(f"no function for caller address {caller_address:#x}")  # pragma: no cover
	return function.name


def _row_name(
	program: Program,
	caller_address: Address,
	dynamic_by_caller: Mapping[Address, tuple[tuple[str, ...], int]],
) -> str:
	name = _caller_name(program, caller_address)
	if name == "<anonymous>" and caller_address in dynamic_by_caller:
		return ", ".join(dynamic_by_caller[caller_address][0])
	return name
