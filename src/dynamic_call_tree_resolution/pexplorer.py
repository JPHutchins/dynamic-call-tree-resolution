# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Typed reading of pexplorer's JSON firmware reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
from msgspec import Struct, field

from dynamic_call_tree_resolution.model import Address

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path


class PexplorerCallee(Struct):
	"""One call edge in pexplorer's report; dynamic edges have no target."""

	call_from: int | None = field(default=None, name="from")
	call_from_function_name: str | None = field(default=None, name="from_function_name")
	call_to: int | None = field(default=None, name="to")
	call_to_function_name: str | None = field(default=None, name="to_function_name")
	dynamic: bool = False


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


def dynamic_sites_by_caller(report: PexplorerReport) -> Mapping[Address, tuple[str, int]]:
	"""Per caller: name and dynamic-call count, keyed by aligned address."""
	return {
		Address(function.address & ~1): (
			function.name,
			sum(callee.dynamic for callee in function.callees),
		)
		for function in report.functions
		if any(callee.dynamic for callee in function.callees)
	}
