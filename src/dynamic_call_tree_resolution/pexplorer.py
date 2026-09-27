# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Typed reading of pexplorer's JSON firmware reports."""

from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING

import msgspec
from msgspec import Struct, field
from salix import Struct as SalixStruct

from dynamic_call_tree_resolution.model import Address, aligned

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path


class PexplorerCallee(Struct):
	"""One call edge in pexplorer's report; dynamic edges have no target."""

	dynamic: bool
	call_from: int | None = field(default=None, name="from")
	call_from_function_name: str | None = field(default=None, name="from_function_name")
	call_to: int | None = field(default=None, name="to")
	call_to_function_name: str | None = field(default=None, name="to_function_name")


class PexplorerFunction(Struct):
	"""A function in pexplorer's report, with its call edges."""

	name: str
	address: int
	callees: tuple[PexplorerCallee, ...] = ()


class PexplorerReport(Struct):
	"""pexplorer's SElfReport JSON, reduced to what the comparison joins on."""

	functions: tuple[PexplorerFunction, ...]


def load_pexplorer(path: Path) -> PexplorerReport:
	"""Parse pexplorer's JSON report from ``path``."""
	return msgspec.json.decode(path.read_bytes(), type=PexplorerReport)


class DynamicSites(SalixStruct):
	"""One caller's aggregated pexplorer dynamic sites."""

	names: tuple[str, ...]
	total: int


def dynamic_sites_by_caller(report: PexplorerReport) -> Mapping[Address, DynamicSites]:
	"""Per caller: names and summed dynamic-call count, keyed by aligned address.

	Functions sharing an aligned address (aliases, ARM/Thumb twins)
	aggregate instead of overwriting, so no caller's dynamic sites are
	dropped by the join.
	"""
	return {
		address: DynamicSites(
			names=tuple(function.name for function in functions),
			total=sum(_dynamic_count(function) for function in functions),
		)
		for address, group in groupby(
			sorted(
				(function for function in report.functions if _dynamic_count(function)),
				key=_aligned_address,
			),
			key=_aligned_address,
		)
		for functions in (tuple(group),)
	}


def _aligned_address(function: PexplorerFunction) -> Address:
	return aligned(Address(function.address))


def _dynamic_count(function: PexplorerFunction) -> int:
	return sum(callee.dynamic for callee in function.callees)
