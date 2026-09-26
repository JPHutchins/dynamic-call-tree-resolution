# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""JSON-serializable analysis reports."""

from typing import TYPE_CHECKING

from msgspec import Struct

from dynamic_call_tree_resolution.call_sites import call_site_candidates
from dynamic_call_tree_resolution.model import FunctionSignature, Provenance, render_path
from dynamic_call_tree_resolution.points_to import unresolved_slots

if TYPE_CHECKING:
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
		resolved_slots=len(resolved),
		resolved_targets=sum(len(assignment.candidates) for assignment in resolved),
	)


def _render_signature(signature: FunctionSignature | None) -> SignatureReport | None:
	return (
		SignatureReport(return_type=signature.return_type, parameters=signature.parameters)
		if signature is not None
		else None
	)
