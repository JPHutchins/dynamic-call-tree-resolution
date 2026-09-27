# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Extraction of indirect call sites from machine code and per-site resolution.

Indirect call and tail-branch instructions are detected and resolved by
the whole-function value-set analysis in :mod:`dynamic_call_tree_resolution.vsa`.
Candidate sets are chased through the loaded image here: a function
address is a target, a slot assignment contributes its candidates, and
pointer chains dereference one slot at a time. An empty result means the
site could not be resolved; consumers fall back to the union of all
resolved targets, keeping stack-depth expansions a sound upper bound.
"""

from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING

from dynamic_call_tree_resolution.model import Address, FunctionSignature
from dynamic_call_tree_resolution.points_to import pointer_at, signatures_by_slot, unresolved_slots
from dynamic_call_tree_resolution.vsa import analyze

if TYPE_CHECKING:
	from collections.abc import Mapping

	from dynamic_call_tree_resolution.model import CallSite, Program, SlotAssignment


def extract_call_sites(program: Program) -> tuple[CallSite, ...]:
	"""Extract every indirect call and tail-branch site from the program's code.

	Sites carry the address their target is taken from when the operand's
	value set is a single address, plus the set of addresses the analysis
	tracked into the operand.
	"""
	return analyze(program)


def matching_targets(program: Program, signature: FunctionSignature) -> frozenset[Address]:
	"""Functions whose DWARF signature exactly matches the given one.

	The loader strips qualifiers when rendering signatures, so equality
	holds across translation units; a cast in the image can violate the
	match, and the consumers' resolved-target fallback covers that.
	"""
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
	"""Candidate target functions of one call site.

	Every address tracked into the site is chased through the loaded
	image; a chase that ends in an unreadable slot with a known signature
	narrows to the matching functions. An empty result means the site
	could not be resolved and consumers fall back to the union of all
	resolved targets.
	"""
	signatures = (
		signatures_by_slot if signatures_by_slot is not None else dict[Address, FunctionSignature]()
	)
	chased = frozenset(
		address
		for candidate in site.candidates
		for address in _chase_target(program, candidate, resolved_by_slot, signatures, frozenset())
	)
	if chased:
		return chased
	signature = signatures.get(site.slot) if site.slot is not None else None
	return matching_targets(program, signature) if signature is not None else frozenset()


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
	target = pointer_at(program, address)
	if target is None:
		signature = signatures_by_slot.get(address)
		return matching_targets(program, signature) if signature is not None else frozenset()
	return _chase_target(program, target, resolved_by_slot, signatures_by_slot, visited | {address})


def per_caller_candidates(
	program: Program,
	sites: tuple[CallSite, ...],
	resolved: tuple[SlotAssignment, ...],
) -> tuple[Mapping[str, frozenset[str]], frozenset[str]]:
	"""Union, per caller, of each site's candidate target names, plus the fallback.

	Unresolved sites contribute the fallback union of all resolved targets,
	so a caller with an unresolved site is indistinguishable from a caller
	with no extracted sites; both keep the expansion a sound upper bound.
	"""
	resolved_map = {assignment.slot: assignment for assignment in resolved}
	signatures = signatures_by_slot(unresolved_slots(program, resolved))
	fallback_addresses = frozenset(
		address for assignment in resolved for address in assignment.candidates
	)

	def caller_name(site: CallSite) -> str:
		return program.functions[site.caller_address].name

	targets_by_caller = {
		caller: frozenset(
			program.functions[address].name
			for site in group
			for address in (
				call_site_candidates(program, site, resolved_map, signatures) or fallback_addresses
			)
		)
		for caller, group in groupby(sorted(sites, key=caller_name), key=caller_name)
	}
	return (
		targets_by_caller,
		frozenset(program.functions[address].name for address in fallback_addresses),
	)
