# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Extraction of indirect call sites from machine code and per-site resolution."""

from __future__ import annotations

from collections import Counter
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, assert_never

from salix import Struct

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
	signatures = (
		signatures_by_slot if signatures_by_slot is not None else dict[Address, FunctionSignature]()
	)
	chased = _chased(program, site, resolved_by_slot, signatures)
	if chased:
		return chased
	signature = signatures.get(site.slot) if site.slot is not None else None
	return matching_targets(program, signature) if signature is not None else frozenset()


def _chased(
	program: Program,
	site: CallSite,
	resolved_by_slot: Mapping[Address, SlotAssignment],
	signatures_by_slot: Mapping[Address, FunctionSignature],
) -> frozenset[Address]:
	match site.target:
		case Known(values=values):
			return frozenset(
				address
				for candidate in values
				for address in _chase_target(
					program, candidate, resolved_by_slot, signatures_by_slot, frozenset()
				)
			)
		case Top() | Unreached():
			return frozenset()
		case _ as unreachable:
			assert_never(unreachable)


def _chase_target(
	program: Program,
	address: Address,
	resolved_by_slot: Mapping[Address, SlotAssignment],
	signatures_by_slot: Mapping[Address, FunctionSignature],
	visited: frozenset[Address],
) -> frozenset[Address]:
	if address in visited:
		return frozenset()
	if address in program.functions:
		return frozenset({address})
	assignment = resolved_by_slot.get(address)
	if assignment is not None:
		return assignment.candidates
	target = None if in_writable_memory(program, address) else pointer_at(program, address)
	if target is None:
		signature = signatures_by_slot.get(address)
		return matching_targets(program, signature) if signature is not None else frozenset()
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


def field_only_targets(
	program: Program,
	sites: tuple[CallSite, ...],
	resolved: tuple[SlotAssignment, ...],
	*,
	narrow_by_signature: bool = False,
	narrowed_by_field: tuple[NarrowedSpan, ...],
	names: Mapping[Address, str] | None = None,
) -> Mapping[str, frozenset[str]]:
	"""The targets each caller reaches only through ``--narrow-by-field``, by caller key."""
	return {
		caller: only
		for caller, group in groupby(
			_caller_targets(
				program, sites, resolved, narrow_by_signature, narrowed_by_field, names
			),
			key=_caller,
		)
		for entries in (tuple(group),)
		for only in (
			frozenset(target for _, target, by_field in entries if by_field)
			- frozenset(target for _, target, by_field in entries if not by_field),
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
) -> tuple[tuple[str, str, bool], ...]:
	resolved_map = {assignment.slot: assignment for assignment in resolved}
	signatures = (
		signatures_by_slot(unresolved_slots(program, resolved)) if narrow_by_signature else None
	)
	name = partial(stack_name, program, names)
	return tuple(
		sorted(
			(frame_key(name(site.caller_address)), target, bool(field_targets))
			for site in sites
			for chased in (call_site_candidates(program, site, resolved_map, signatures),)
			for field_targets in (
				frozenset[Address]()
				if chased
				else narrowed(frozenset[Address](), narrowed_by_field, site.site_address),
			)
			for target in frozenset(map(name, chased or field_targets)) or {INDIRECT_CALLEE}
		)
	)


def _caller(entry: tuple[str, str, bool]) -> str:
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
		field_targets_by_caller=field_only_targets(
			program,
			thread.sites,
			resolved,
			narrow_by_signature=narrow_by_signature,
			narrowed_by_field=narrowed_by_field,
			names=names,
		),
	)
