# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The descriptors plugin's records, from the alias dumps a build's GCC wrote (#205)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from salix import Struct

from dynamic_call_tree_resolution.gcc_dumps.dump import dump_functions
from dynamic_call_tree_resolution.gcc_dumps.records import (
	Gap,
	SiteRecord,
	StoreRecord,
	record_line,
	unit_records,
)
from dynamic_call_tree_resolution.gcc_dumps.units import DUMP_FLAG as DUMP_FLAG
from dynamic_call_tree_resolution.gcc_dumps.units import dumped_units, unit_types

if TYPE_CHECKING:
	from pathlib import Path


class DumpDescriptors(Struct):
	"""A build's descriptor records, and the statements no record could be made of."""

	lines: tuple[str, ...]
	"""Sorted, as the plugin's replay writes ``descriptors.txt``."""
	gaps: tuple[Gap, ...]
	units: int
	"""The build's C units."""
	dumps: int
	"""Those GCC wrote an alias dump for: a unit without a function, or not optimized, has none."""


def dump_descriptors(build_directory: Path, root: Path) -> DumpDescriptors:
	units = dumped_units(build_directory)
	records = tuple(
		record
		for unit in units
		if unit.dump is not None
		for record in unit_records(
			unit.unit, dump_functions(unit.dump.read_text()), unit_types(unit.object), root
		)
	)
	return DumpDescriptors(
		units=len(units),
		dumps=sum(unit.dump is not None for unit in units),
		lines=tuple(
			sorted(
				record_line(record)
				for record in records
				if isinstance(record, SiteRecord | StoreRecord)
			)
		),
		gaps=tuple(record for record in records if isinstance(record, Gap)),
	)
