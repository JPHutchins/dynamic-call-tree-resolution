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
from dynamic_call_tree_resolution.stack_analysis import (
	INDIRECT_CALLEE,
	NULL_CALLEE,
	ThreadTargets,
	frame_key,
)
from dynamic_call_tree_resolution.vsa import Analysis, address_taken, analyze, runtime_value
from dynamic_call_tree_resolution.vsa.lattice import Known, Top
from dynamic_call_tree_resolution.vsa.vectors import hardware_handlers

if TYPE_CHECKING:
	from collections.abc import Callable, Mapping

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
	fallback: frozenset[Address]


def resolve(program: Program, rtos: RtosModel = BARE_METAL) -> ProgramResolution:
	analysis = analyze(program, rtos)
	return ProgramResolution(
		sites=analysis.sites,
		seeded=analysis.seeded,
		threads=analysis.threads,
		fallback=_fallback(program, analysis.address_taken),
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
	narrowings: Mapping[Address, SignatureNarrowed] | None = None,
) -> frozenset[Address]:
	return frozenset(
		target
		for found in site_targets(program, site, resolved_by_slot, narrowings)
		for target in target_addresses(found)
	)


class Chased(Struct):
	"""What the image's values say a site can call."""

	targets: frozenset[Address]


class SignatureNarrowed(Struct):
	"""The functions of the site's slot's signature, as ``--narrow-by-signature`` assumes."""

	signature: FunctionSignature
	targets: frozenset[Address]


class External(Struct):
	"""A call through a slot the dynamic loader fills with an undefined symbol's address."""

	symbol: str


class Null(Struct):
	"""A call whose target can only be address 0, which no function starts at."""


type SiteTargets = Chased | SignatureNarrowed | External | Null


def target_addresses(found: SiteTargets) -> frozenset[Address]:
	match found:
		case Chased(targets=targets) | SignatureNarrowed(targets=targets):
			return targets
		case External() | Null():
			return frozenset()
		case _ as unreachable:
			assert_never(unreachable)


def fallback_addresses(program: Program) -> frozenset[Address]:
	return _fallback(program, address_taken(program))


def _fallback(program: Program, taken: frozenset[Address]) -> frozenset[Address]:
	return taken - hardware_handlers(program)


def signature_narrowings(
	program: Program,
	signatures: Mapping[Address, FunctionSignature],
	fallback: frozenset[Address] | None = None,
) -> Mapping[Address, SignatureNarrowed]:
	return {
		slot: SignatureNarrowed(
			signature=signature, targets=matching_targets(program, signature) & available
		)
		for available in (fallback if fallback is not None else fallback_addresses(program),)
		for slot, signature in signatures.items()
	}


def site_targets(
	program: Program,
	site: CallSite,
	resolved_by_slot: Mapping[Address, SlotAssignment],
	narrowings: Mapping[Address, SignatureNarrowed] | None = None,
) -> tuple[SiteTargets, ...]:
	"""What a site can call, each set with where it came from."""
	if site.external is not None:
		return (External(symbol=site.external),)
	if _null(program, site):
		return (Null(),)
	by_slot = narrowings if narrowings is not None else dict[Address, SignatureNarrowed]()
	chased = _chased(program, site, resolved_by_slot, by_slot)
	return (
		chased
		if any(map(target_addresses, chased))
		else tuple(
			narrowing
			for narrowing in (
				by_slot.get(site.loaded_from) if site.loaded_from is not None else None,
			)
			if narrowing is not None
		)
	)


def _null(program: Program, site: CallSite) -> bool:
	return site.target == Known(values=frozenset({Address(0)})) and (
		site.loaded_from is None or not in_writable_memory(program, site.loaded_from)
	)


def narrowing_signature(findings: tuple[SiteTargets, ...]) -> FunctionSignature | None:
	return next(
		(
			signature
			for found in findings
			for signature in (_signature(found),)
			if signature is not None and target_addresses(found)
		),
		None,
	)


def _signature(found: SiteTargets) -> FunctionSignature | None:
	match found:
		case Chased() | External() | Null():
			return None
		case SignatureNarrowed(signature=signature):
			return signature
		case _ as unreachable:
			assert_never(unreachable)


def _chased(
	program: Program,
	site: CallSite,
	resolved_by_slot: Mapping[Address, SlotAssignment],
	narrowings: Mapping[Address, SignatureNarrowed],
) -> tuple[SiteTargets, ...]:
	match site.target:
		case Known(values=values):
			return tuple(
				_chase_target(program, candidate, resolved_by_slot, narrowings, frozenset())
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
	narrowings: Mapping[Address, SignatureNarrowed],
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
		return narrowings.get(address, Chased(targets=frozenset()))
	return _chase_target(program, target, resolved_by_slot, narrowings, visited | {address})


def per_caller_candidates(
	program: Program,
	sites: tuple[CallSite, ...],
	resolved: tuple[SlotAssignment, ...],
	*,
	narrow_by_signature: bool = False,
	narrowed_by_field: tuple[NarrowedSpan, ...] = (),
	names: Mapping[Address, str] | None = None,
	fallback: frozenset[Address] | None = None,
) -> tuple[Mapping[str, frozenset[str]], frozenset[str]]:
	available = fallback if fallback is not None else fallback_addresses(program)
	return (
		_targets_by_caller(
			program,
			sites,
			resolved,
			narrow_by_signature=narrow_by_signature,
			narrowed_by_field=narrowed_by_field,
			names=names,
			fallback=available,
		),
		frozenset(map(partial(stack_name, program, names), available)),
	)


def _targets_by_caller(
	program: Program,
	sites: tuple[CallSite, ...],
	resolved: tuple[SlotAssignment, ...],
	*,
	narrow_by_signature: bool,
	narrowed_by_field: tuple[NarrowedSpan, ...],
	names: Mapping[Address, str] | None,
	fallback: frozenset[Address] | None,
) -> Mapping[str, frozenset[str]]:
	return {
		caller: frozenset(target for _, target, _ in group)
		for caller, group in groupby(
			_caller_targets(
				program, sites, resolved, narrow_by_signature, narrowed_by_field, names, fallback
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
	fallback: frozenset[Address] | None = None,
) -> Mapping[str, Mapping[str, EdgeKind]]:
	"""The targets each caller reaches only through a narrowing, with its edge, by caller key."""
	return {
		caller: only
		for caller, group in groupby(
			_caller_targets(
				program, sites, resolved, narrow_by_signature, narrowed_by_field, names, fallback
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
	fallback: frozenset[Address] | None,
) -> tuple[tuple[str, str, EdgeKind], ...]:
	resolved_map = {assignment.slot: assignment for assignment in resolved}
	narrowings = (
		signature_narrowings(
			program, signatures_by_slot(unresolved_slots(program, resolved)), fallback
		)
		if narrow_by_signature
		else None
	)
	name = partial(stack_name, program, names)
	return tuple(
		sorted(
			(frame_key(name(site.caller_address)), target, kind)
			for site in sites
			for targets, kind in _labeled(
				site_targets(program, site, resolved_map, narrowings),
				narrowed(frozenset[Address](), narrowed_by_field, site.site_address),
				name,
			)
			for target in targets or {INDIRECT_CALLEE}
		)
	)


def _labeled(
	findings: tuple[SiteTargets, ...],
	field: frozenset[Address],
	name: Callable[[Address], str],
) -> tuple[tuple[frozenset[str], EdgeKind], ...]:
	return tuple(
		(names, _kind(found)) for found in findings for names in (_names(found, name),) if names
	) or (
		((frozenset(map(name, field)), EdgeKind.FIELD),)
		if field
		else ((frozenset[str](), EdgeKind.CANDIDATE),)
	)


def _names(found: SiteTargets, name: Callable[[Address], str]) -> frozenset[str]:
	match found:
		case Chased(targets=targets) | SignatureNarrowed(targets=targets):
			return frozenset(map(name, targets))
		case External(symbol=symbol):
			return frozenset({symbol})
		case Null():
			return frozenset({NULL_CALLEE})
		case _ as unreachable:
			assert_never(unreachable)


def _kind(found: SiteTargets) -> EdgeKind:
	match found:
		case Chased():
			return EdgeKind.CANDIDATE
		case SignatureNarrowed():
			return EdgeKind.SIGNATURE
		case External():
			return EdgeKind.EXTERNAL
		case Null():
			return EdgeKind.CANDIDATE
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
	fallback: frozenset[Address] | None = None,
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
			fallback=fallback,
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
			fallback=fallback,
		),
	)
