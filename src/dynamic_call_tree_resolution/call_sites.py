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

from typing import TYPE_CHECKING

from dynamic_call_tree_resolution.points_to import pointer_at
from dynamic_call_tree_resolution.vsa import analyze

if TYPE_CHECKING:
	from collections.abc import Mapping

	from dynamic_call_tree_resolution.model import Address, CallSite, Program, SlotAssignment


def extract_call_sites(program: Program) -> tuple[CallSite, ...]:
	"""Extract every indirect call and tail-branch site from the program's code.

	Sites carry the address their target is taken from when the operand's
	value set is a single address, plus the set of addresses the analysis
	tracked into the operand. Unsupported machine types yield no sites.
	"""
	return analyze(program)


def call_site_candidates(
	program: Program, site: CallSite, resolved_by_slot: Mapping[Address, SlotAssignment]
) -> frozenset[Address]:
	"""Candidate target functions of one call site.

	Every address tracked into the site is chased through the loaded
	image; an empty result means the site could not be resolved and
	consumers fall back to the union of all resolved targets.
	"""
	return frozenset(
		address
		for candidate in site.candidates
		for address in _chase_target(program, candidate, resolved_by_slot, frozenset())
	)


def _chase_target(
	program: Program,
	address: Address,
	resolved_by_slot: Mapping[Address, SlotAssignment],
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
		return frozenset()
	return _chase_target(program, target, resolved_by_slot, visited | {address})


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
	resolved_by_slot = {assignment.slot: assignment for assignment in resolved}
	fallback_addresses = frozenset(
		address for assignment in resolved for address in assignment.candidates
	)
	by_caller: dict[str, set[str]] = {}
	for site in sites:
		caller = program.functions[site.caller_address].name
		candidates = call_site_candidates(program, site, resolved_by_slot) or fallback_addresses
		by_caller.setdefault(caller, set()).update(
			program.functions[address].name for address in candidates
		)
	return (
		{caller: frozenset(targets) for caller, targets in by_caller.items()},
		frozenset(program.functions[address].name for address in fallback_addresses),
	)
