# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""JSON-serializable analysis reports."""

from typing import TYPE_CHECKING

from msgspec import Struct

from dynamic_call_tree_resolution.model import Provenance, render_path

if TYPE_CHECKING:
	from dynamic_call_tree_resolution.model import Program, SlotAssignment


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


class AnalysisReport(Struct):
	"""All resolved slots of one program, with aggregate counts."""

	assignments: tuple[SlotAssignmentReport, ...]
	resolved_slots: int
	resolved_targets: int


class AnalysisSummary(Struct):
	"""Cross-cutting resolution and stack summary for CI reporting."""

	resolved_slots: int
	resolved_targets: int
	indirect_call_sites: int
	total_functions: int
	entry_points: int
	worst_case_bytes: int
	worst_case_entry: str


def build_report(program: Program, resolved: tuple[SlotAssignment, ...]) -> AnalysisReport:
	"""Render resolved assignments for JSON export."""
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
		resolved_slots=len(resolved),
		resolved_targets=sum(len(assignment.candidates) for assignment in resolved),
	)
