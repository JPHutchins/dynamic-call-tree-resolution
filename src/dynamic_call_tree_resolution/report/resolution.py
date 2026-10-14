# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The report of what an image's indirect calls resolve to."""

from typing import TYPE_CHECKING, Final

from msgspec import Struct
from salix import Struct as SalixStruct

from dynamic_call_tree_resolution.call_sites import (
	Dispatched,
	Null,
	calls_nothing,
	narrowing_signature,
	signature_narrowings,
	site_targets,
	target_addresses,
)
from dynamic_call_tree_resolution.field_narrowing import narrowing_at
from dynamic_call_tree_resolution.model import (
	BARE_METAL,
	Address,
	Dead,
	FunctionSignature,
	NullSlot,
	Provenance,
	Residue,
	SkipReason,
	UnresolvedSlot,
	render_path,
)
from dynamic_call_tree_resolution.points_to import (
	not_enumerated,
	null_slots,
	signatures_by_slot,
	unresolved_slots,
)

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
