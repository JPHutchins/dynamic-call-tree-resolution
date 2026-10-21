# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Where each address-taken function's address is held or computed."""

from itertools import chain
from typing import TYPE_CHECKING

from msgspec import Struct

from dynamic_call_tree_resolution.vsa import address_taken, referrers
from dynamic_call_tree_resolution.vsa.links import function_covering

if TYPE_CHECKING:
	from dynamic_call_tree_resolution.model import Address, Program


class ReferrerReport(Struct):
	"""One place that holds or computes an address-taken function's address."""

	slot: int
	holder: str | None
	"""The function or data object spanning the slot; none outside every symbol."""


class AddressTakenReport(Struct):
	"""One row of ``referrers --json``."""

	name: str
	address: int
	referrers: tuple[ReferrerReport, ...]


def referrers_report(program: Program) -> tuple[AddressTakenReport, ...]:
	return tuple(
		sorted(
			(
				AddressTakenReport(
					name=program.functions[function].name,
					address=function,
					referrers=tuple(
						ReferrerReport(slot=slot, holder=_holder(program, slot))
						for slot in sorted(slots)
					),
				)
				for function, slots in referrers(program, address_taken(program)).items()
			),
			key=_name_and_address,
		)
	)


class _NameAndAddress(Struct, order=True):
	"""Where a report sorts among reports."""

	name: str
	address: int


def _name_and_address(report: AddressTakenReport) -> _NameAndAddress:
	return _NameAndAddress(name=report.name, address=report.address)


def _holder(program: Program, slot: Address) -> str | None:
	return next(
		chain(
			(
				function.name
				for function in (function_covering(program, slot),)
				if function is not None
			),
			(
				data_object.name
				for data_object in program.objects.values()
				if slot - data_object.address in range(max(data_object.size, 1))
			),
		),
		None,
	)
