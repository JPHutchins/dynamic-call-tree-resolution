# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Extraction of indirect call sites from machine code and per-site resolution."""

from __future__ import annotations

from collections import Counter
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, assert_never

from salix import Struct

from dynamic_call_tree_resolution.callgraph import EdgeKind
from dynamic_call_tree_resolution.field_narrowing import narrowed
from dynamic_call_tree_resolution.identity import stack_name
from dynamic_call_tree_resolution.model import (
	BARE_METAL,
	Address,
	FunctionSignature,
	Provenance,
	SlotAssignment,
	Unreached,
)
from dynamic_call_tree_resolution.points_to import (
	assignments,
	in_writable_memory,
	pointer_at,
	signatures_by_slot,
	unresolved_slots,
)
from dynamic_call_tree_resolution.stack_analysis import INDIRECT_CALLEE, ThreadTargets, frame_key
from dynamic_call_tree_resolution.vsa import Analysis, address_taken, analyze, runtime_value
from dynamic_call_tree_resolution.vsa.lattice import Known, Top
from dynamic_call_tree_resolution.vsa.vectors import hardware_handlers

if TYPE_CHECKING:
	from collections.abc import Mapping

	from dynamic_call_tree_resolution.field_narrowing import NarrowedSpan
	from dynamic_call_tree_resolution.model import CallSite, Program, RtosModel
	from dynamic_call_tree_resolution.vsa.analysis import ThreadSites


def extract_call_sites(program: Program) -> tuple[CallSite, ...]:
	return analyze(program).sites


class ProgramResolution(Struct):
	"""One image's call sites and the slot assignments that hold at runtime."""

	sites: tuple[CallSite, ...]
	assignments: tuple[SlotAssignment, ...]
	"""RAM initializers include the program's stores, and drop out when a store is unknown."""
	seeded: frozenset[str]
	threads: Mapping[str, ThreadSites]


def resolve(program: Program, rtos: RtosModel = BARE_METAL) -> ProgramResolution:
	analysis = analyze(program, rtos)
	return ProgramResolution(
		sites=analysis.sites,
		seeded=analysis.seeded,
		threads=analysis.threads,
		assignments=tuple(
			runtime
			for assignment in assignments(program)
			if (runtime := _at_runtime(program, analysis, assignment)) is not None
		),
	)


def _at_runtime(
	program: Program, analysis: Analysis, assignment: SlotAssignment
) -> SlotAssignment | None:
	match assignment.provenance:
		case Provenance.ROM_CONSTANT:
			return assignment
		case Provenance.RAM_INITIALIZER:
			match runtime_value(analysis.context, assignment.slot):
				case Top():
					return None
				case Known(values=values):
					return (
						SlotAssignment(
							slot=assignment.slot,
							path=assignment.path,
							candidates=values - {Address(0)},
							provenance=assignment.provenance,
							relocated=assignment.relocated,
						)
						if all(value == 0 or value in program.functions for value in values)
						else None
					)
				case _ as unreachable:
					assert_never(unreachable)
		case _ as unreachable:
			assert_never(unreachable)


def matching_targets(program: Program, signature: FunctionSignature) -> frozenset[Address]:
	return frozenset(
		function.address
		for function in program.functions.values()
		if function.signature == signature
	)


def call_site_candidates(
	program: Program,
	site: CallSite,
	resolved_by_slot: Mapping[Address, SlotAssignment],
	signatures_by_slot: Mapping[Address, FunctionSignature] | None = None,
) -> frozenset[Address]:
	return frozenset(
		target
		for found in site_targets(program, site, resolved_by_slot, signatures_by_slot)
		for target in found.targets
	)


class Chased(Struct):
	"""What the image's values say a site can call."""

	targets: frozenset[Address]


class SignatureNarrowed(Struct):
	"""The functions of the site's slot's signature, as ``--narrow-by-signature`` assumes."""

	signature: FunctionSignature
	targets: frozenset[Address]


type SiteTargets = Chased | SignatureNarrowed


def site_targets(
	program: Program,
	site: CallSite,
	resolved_by_slot: Mapping[Address, SlotAssignment],
	signatures_by_slot: Mapping[Address, FunctionSignature] | None = None,
) -> tuple[SiteTargets, ...]:
	"""What a site can call, each set with where it came from."""
	signatures = (
		signatures_by_slot if signatures_by_slot is not None else dict[Address, FunctionSignature]()
	)
	chased = _chased(program, site, resolved_by_slot, signatures)
	return (
		chased
		if any(found.targets for found in chased)
		else tuple(
			_narrowed_by(program, signature)
			for signature in (signatures.get(site.slot) if site.slot is not None else None,)
			if signature is not None
		)
	)


def narrowing_signature(findings: tuple[SiteTargets, ...]) -> FunctionSignature | None:
	return next(
		(
			signature
			for found in findings
			for signature in (_signature(found),)
			if signature is not None and found.targets
		),
		None,
	)


def _signature(found: SiteTargets) -> FunctionSignature | None:
	match found:
		case Chased():
			return None
		case SignatureNarrowed(signature=signature):
			return signature
		case _ as unreachable:
			assert_never(unreachable)


def _narrowed_by(program: Program, signature: FunctionSignature) -> SiteTargets:
	return SignatureNarrowed(signature=signature, targets=matching_targets(program, signature))


def _chased(
	program: Program,
	site: CallSite,
	resolved_by_slot: Mapping[Address, SlotAssignment],
	signatures_by_slot: Mapping[Address, FunctionSignature],
) -> tuple[SiteTargets, ...]:
	match site.target:
		case Known(values=values):
			return tuple(
				_chase_target(program, candidate, resolved_by_slot, signatures_by_slot, frozenset())
				for candidate in sorted(values)
			)
		case Top() | Unreached():
			return ()
		case _ as unreachable:
			assert_never(unreachable)


def _chase_target(
	program: Program,
	address: Address,
	resolved_by_slot: Mapping[Address, SlotAssignment],
	signatures_by_slot: Mapping[Address, FunctionSignature],
	visited: frozenset[Address],
) -> SiteTargets:
	if address in visited:
		return Chased(targets=frozenset())
	if address in program.functions:
		return Chased(targets=frozenset({address}))
	assignment = resolved_by_slot.get(address)
	if assignment is not None:
		return Chased(targets=assignment.candidates)
	target = None if in_writable_memory(program, address) else pointer_at(program, address)
	if target is None:
		signature = signatures_by_slot.get(address)
		return (
			_narrowed_by(program, signature)
			if signature is not None
			else Chased(targets=frozenset())
		)
	return _chase_target(program, target, resolved_by_slot, signatures_by_slot, visited | {address})


def per_caller_candidates(
	program: Program,
	sites: tuple[CallSite, ...],
	resolved: tuple[SlotAssignment, ...],
	*,
	narrow_by_signature: bool = False,
	narrowed_by_field: tuple[NarrowedSpan, ...] = (),
	names: Mapping[Address, str] | None = None,
) -> tuple[Mapping[str, frozenset[str]], frozenset[str]]:
	return (
		_targets_by_caller(
			program,
			sites,
			resolved,
			narrow_by_signature=narrow_by_signature,
			narrowed_by_field=narrowed_by_field,
			names=names,
		),
		frozenset(
			map(
				partial(stack_name, program, names),
				address_taken(program) - hardware_handlers(program),
			)
		),
	)


def _targets_by_caller(
	program: Program,
	sites: tuple[CallSite, ...],
	resolved: tuple[SlotAssignment, ...],
	*,
	narrow_by_signature: bool,
	narrowed_by_field: tuple[NarrowedSpan, ...],
	names: Mapping[Address, str] | None,
) -> Mapping[str, frozenset[str]]:
	return {
		caller: frozenset(target for _, target, _ in group)
		for caller, group in groupby(
			_caller_targets(
				program, sites, resolved, narrow_by_signature, narrowed_by_field, names
			),
			key=_caller,
		)
	}


def narrowed_targets(
	program: Program,
	sites: tuple[CallSite, ...],
	resolved: tuple[SlotAssignment, ...],
	*,
	narrow_by_signature: bool = False,
	narrowed_by_field: tuple[NarrowedSpan, ...] = (),
	names: Mapping[Address, str] | None = None,
) -> Mapping[str, Mapping[str, EdgeKind]]:
	"""The targets each caller reaches only through a narrowing, with its edge, by caller key."""
	return {
		caller: only
		for caller, group in groupby(
			_caller_targets(
				program, sites, resolved, narrow_by_signature, narrowed_by_field, names
			),
			key=_caller,
		)
		for entries in (tuple(group),)
		for candidates in (
			frozenset(target for _, target, kind in entries if kind is EdgeKind.CANDIDATE),
		)
		for only in (
			{
				target: kind
				for _, target, kind in entries
				if kind is not EdgeKind.CANDIDATE and target not in candidates
			},
		)
		if only
	}


def _caller_targets(
	program: Program,
	sites: tuple[CallSite, ...],
	resolved: tuple[SlotAssignment, ...],
	narrow_by_signature: bool,
	narrowed_by_field: tuple[NarrowedSpan, ...],
	names: Mapping[Address, str] | None,
) -> tuple[tuple[str, str, EdgeKind], ...]:
	resolved_map = {assignment.slot: assignment for assignment in resolved}
	signatures = (
		signatures_by_slot(unresolved_slots(program, resolved)) if narrow_by_signature else None
	)
	name = partial(stack_name, program, names)
	return tuple(
		sorted(
			(frame_key(name(site.caller_address)), target, kind)
			for site in sites
			for targets, kind in _labeled(
				site_targets(program, site, resolved_map, signatures),
				narrowed(frozenset[Address](), narrowed_by_field, site.site_address),
			)
			for target in frozenset(map(name, targets)) or {INDIRECT_CALLEE}
		)
	)


def _labeled(
	findings: tuple[SiteTargets, ...], field: frozenset[Address]
) -> tuple[tuple[frozenset[Address], EdgeKind], ...]:
	return tuple((found.targets, _kind(found)) for found in findings if found.targets) or (
		((field, EdgeKind.FIELD),) if field else ((frozenset[Address](), EdgeKind.CANDIDATE),)
	)


def _kind(found: SiteTargets) -> EdgeKind:
	match found:
		case Chased():
			return EdgeKind.CANDIDATE
		case SignatureNarrowed():
			return EdgeKind.SIGNATURE
		case _ as unreachable:
			assert_never(unreachable)


def _caller(entry: tuple[str, str, EdgeKind]) -> str:
	return entry[0]


def own_targets(
	program: Program,
	thread: ThreadSites,
	resolved: tuple[SlotAssignment, ...],
	names: Mapping[Address, str] | None = None,
	narrowed_by_field: tuple[NarrowedSpan, ...] = (),
	*,
	narrow_by_signature: bool = False,
) -> ThreadTargets:
	return ThreadTargets(
		reached=frozenset(
			frame_key(stack_name(program, names, address)) for address in thread.reached
		),
		targets_by_caller=_targets_by_caller(
			program,
			thread.sites,
			resolved,
			narrow_by_signature=narrow_by_signature,
			narrowed_by_field=narrowed_by_field,
			names=names,
		),
		sites_by_caller=Counter(
			frame_key(stack_name(program, names, site.caller_address)) for site in thread.sites
		),
		narrowed_by_caller=narrowed_targets(
			program,
			thread.sites,
			resolved,
			narrow_by_signature=narrow_by_signature,
			narrowed_by_field=narrowed_by_field,
			names=names,
		),
	)
