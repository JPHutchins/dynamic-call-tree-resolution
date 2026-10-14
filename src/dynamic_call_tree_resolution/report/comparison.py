# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""A per-function rollup of resolution, for comparison across tools."""

from itertools import groupby
from typing import TYPE_CHECKING

from msgspec import Struct

from dynamic_call_tree_resolution.call_sites import (
	call_site_candidates,
	calls_nothing,
	resolve,
	signature_narrowings,
)
from dynamic_call_tree_resolution.field_narrowing import narrowed
from dynamic_call_tree_resolution.model import (
	BARE_METAL,
	Address,
	Machine,
	aligned,
	thumb_twin,
)
from dynamic_call_tree_resolution.pexplorer import (
	DynamicSites,
	PexplorerReport,
	dynamic_sites_by_caller,
)
from dynamic_call_tree_resolution.points_to import (
	null_slots,
	signatures_by_slot,
	unresolved_slots,
)
from dynamic_call_tree_resolution.report.resolution import (
	resolved_by_slot,
	slot_counts,
)

if TYPE_CHECKING:
	from collections.abc import Mapping

	from dynamic_call_tree_resolution.field_narrowing import NarrowedSpan
	from dynamic_call_tree_resolution.model import CallSite, Program, RtosModel


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
	rtos: str = BARE_METAL.name


def build_comparison(
	name: str,
	program: Program,
	pexplorer: PexplorerReport | None = None,
	*,
	narrow_by_signature: bool = False,
	narrowed_by_field: tuple[NarrowedSpan, ...] = (),
	rtos: RtosModel = BARE_METAL,
) -> ComparisonReport:
	resolution = resolve(program, rtos)
	resolved = resolution.assignments
	unresolved = unresolved_slots(program, resolved)
	resolved_map = resolved_by_slot(resolved)
	narrowings = (
		signature_narrowings(program, signatures_by_slot(unresolved), resolution.fallback)
		if narrow_by_signature
		else None
	)
	counts = slot_counts(resolved, unresolved, null_slots(program, resolved))
	outcomes = tuple(
		_SiteOutcome(
			caller=_site_address(site),
			candidates=0
			if nothing
			else len(
				narrowed(
					call_site_candidates(program, site, resolved_map, narrowings),
					narrowed_by_field,
					site.site_address,
				)
			),
			calls_nothing=nothing,
		)
		for site in resolution.sites
		for nothing in (calls_nothing(program, site),)
	)
	candidate_sizes = sorted(outcome.candidates for outcome in outcomes)
	dynamic_by_caller: Mapping[Address, DynamicSites] = (
		dynamic_sites_by_caller(pexplorer) if pexplorer is not None else {}
	)
	sites_by_caller = {
		caller_address: tuple(group)
		for caller_address, group in groupby(
			sorted(outcomes, key=_outcome_caller), key=_outcome_caller
		)
	}
	rows = (
		*(
			FunctionComparison(
				address=caller_address,
				caller=_caller_name(program, caller_address),
				pexplorer_dynamic_sites=dynamic_by_caller[caller_address].total
				if caller_address in dynamic_by_caller
				else 0,
				dctr_call_sites=len(group),
				dctr_resolved_sites=sum(map(_resolved, group)),
				dctr_exact_sites=sum(map(_exact, group)),
			)
			for caller_address, group in sites_by_caller.items()
		),
		*(
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
		),
	)
	return ComparisonReport(
		elf=name,
		machine=program.machine,
		functions=len(program.functions),
		total_slots=counts.total_slots,
		resolved_slots=counts.resolved_slots,
		unresolved_slots=len(unresolved),
		call_sites=len(candidate_sizes),
		resolved_call_sites=sum(map(_resolved, outcomes)),
		exact_call_sites=sum(map(_exact, outcomes)),
		candidate_size_counts=tuple(
			(size, candidate_sizes.count(size)) for size in dict.fromkeys(candidate_sizes, 0)
		),
		pexplorer_dynamic_sites=(
			sum(sites.total for sites in dynamic_by_caller.values())
			if pexplorer is not None
			else None
		),
		function_comparisons=tuple(sorted(rows, key=_row_caller)),
		rtos=rtos.name,
	)


def _site_address(site: CallSite) -> Address:
	return aligned(site.caller_address)


def _row_caller(row: FunctionComparison) -> str:
	return row.caller


class _SiteOutcome(Struct):
	"""What a comparison counts of one call site."""

	caller: Address
	candidates: int
	calls_nothing: bool


def _outcome_caller(outcome: _SiteOutcome) -> Address:
	return outcome.caller


def _resolved(outcome: _SiteOutcome) -> bool:
	return outcome.candidates > 0 or outcome.calls_nothing


def _exact(outcome: _SiteOutcome) -> bool:
	return outcome.candidates == 1 or outcome.calls_nothing


def _caller_name(program: Program, caller_address: Address) -> str:
	function = program.functions.get(caller_address) or program.functions.get(
		thumb_twin(caller_address)
	)
	if function is None:
		raise ValueError(f"no function for caller address {caller_address:#x}")  # pragma: no cover
	return function.name
