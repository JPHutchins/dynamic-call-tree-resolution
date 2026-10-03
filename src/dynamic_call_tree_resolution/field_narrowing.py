# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Narrowing an indirect call to what its field can hold, assuming no cast writes it (#159)."""

from __future__ import annotations

from bisect import bisect_right
from itertools import groupby
from typing import TYPE_CHECKING, assert_never

from salix import Struct

from dynamic_call_tree_resolution.descriptors import (
	Element,
	Field,
	Function,
	Null,
	Other,
	Parameter,
	Variable,
)
from dynamic_call_tree_resolution.points_to import field_slots, pointer_at
from dynamic_call_tree_resolution.vsa.abi import normalized
from dynamic_call_tree_resolution.vsa.links import function_covering

if TYPE_CHECKING:
	from collections.abc import Mapping

	from dynamic_call_tree_resolution.descriptors import Descriptor, Descriptors, Place, Site, Store
	from dynamic_call_tree_resolution.model import Address, LineSpan, Program, SourceLocation


class FieldNarrowing(Struct):
	"""What a site's field can hold: its initializers and the functions stored into it."""

	field: Field
	targets: frozenset[Address]


class NarrowedSpan(Struct):
	"""Code whose line places it at a call through a narrowable field."""

	start: Address
	end: Address
	narrowing: FieldNarrowing


def field_narrowings(
	program: Program, descriptors: Descriptors, spans: tuple[LineSpan, ...]
) -> tuple[NarrowedSpan, ...]:
	return tuple(
		sorted(
			(
				NarrowedSpan(
					start=span.start,
					end=span.end,
					narrowing=FieldNarrowing(field=field, targets=targets[field]),
				)
				for targets in (field_targets(program, descriptors.stores),)
				for callees in (_callees(descriptors.sites),)
				for names in (_names_by_address(program),)
				for span in spans
				for function in (function_covering(program, span.start),)
				if function is not None
				for field in (
					_site_field(
						callees,
						names.get(normalized(function.address, program.machine), frozenset()),
						span.location,
					),
				)
				if field is not None and field in targets
			),
			key=_span_start,
		)
	)


def narrowing_at(spans: tuple[NarrowedSpan, ...], address: Address) -> FieldNarrowing | None:
	match spans[bisect_right(spans, address, key=_span_start) - 1 :][:1]:
		case (NarrowedSpan(start=start, end=end, narrowing=narrowing),) if start <= address < end:
			return narrowing
		case _:
			return None


def narrowed(
	candidates: frozenset[Address], spans: tuple[NarrowedSpan, ...], address: Address
) -> frozenset[Address]:
	"""The candidates, or, when there are none, what the site's field holds."""
	match candidates or narrowing_at(spans, address):
		case frozenset():
			return candidates
		case None:
			return frozenset()
		case FieldNarrowing(targets=targets):
			return targets
		case _ as unreachable:
			assert_never(unreachable)


def _span_start(span: NarrowedSpan) -> Address:
	return span.start


def field_targets(
	program: Program, stores: tuple[Store, ...]
) -> Mapping[Field, frozenset[Address]]:
	"""Each field that holds only known functions, and at least one.

	A field holds what its instances in the image are initialized to and what every store
	into it stores. A store of anything but a function or a null leaves the field unknown.
	"""
	stored = _stored(stores)
	slots = {
		Field(record=record, member=member): slots
		for (record, member), slots in field_slots(program).items()
	}
	return {
		field: targets
		for functions in (
			{normalized(address, program.machine): address for address in program.functions},
		)
		for field in slots.keys() | stored.keys()
		for targets in (
			_held(
				program,
				functions,
				slots.get(field, frozenset()),
				stored.get(field, frozenset()),
			),
		)
		if targets
	}


def _held(
	program: Program,
	functions: Mapping[Address, Address],
	slots: frozenset[Address],
	stored: frozenset[Descriptor],
) -> frozenset[Address]:
	initial = frozenset(value for slot in slots for value in (pointer_at(program, slot),) if value)
	values = tuple(_stored_functions(program, functions, value) for value in stored)
	return (
		initial.union(*(value for value in values if value is not None))
		if None not in values and initial <= program.functions.keys()
		else frozenset()
	)


def _stored_functions(
	program: Program, functions: Mapping[Address, Address], value: Descriptor
) -> frozenset[Address] | None:
	match value:
		case Function(symbol=symbol):
			starts = program.symbol_addresses.get(symbol, frozenset())
			return (
				frozenset(functions[start] for start in starts)
				if starts and starts <= functions.keys()
				else None
			)
		case Null():
			return frozenset()
		case Field() | Variable() | Parameter() | Element() | Other():
			return None
		case _ as unreachable:
			assert_never(unreachable)


def _stored(stores: tuple[Store, ...]) -> Mapping[Field, frozenset[Descriptor]]:
	return {
		field: frozenset(value for _, value in group)
		for field, group in groupby(
			sorted(
				(
					(field, store.value)
					for store in stores
					for field in (_stored_field(store.place),)
					if field is not None
				),
				key=_field_order,
			),
			key=_stored_into,
		)
	}


def _stored_field(place: Place) -> Field | None:
	match place:
		case Field():
			return place
		case Variable():
			return None
		case _ as unreachable:
			assert_never(unreachable)


def _field_order(entry: tuple[Field, Descriptor]) -> tuple[str, str]:
	return (entry[0].record, entry[0].member)


def _stored_into(entry: tuple[Field, Descriptor]) -> Field:
	return entry[0]


def _callees(sites: tuple[Site, ...]) -> Mapping[tuple[str, SourceLocation], frozenset[Descriptor]]:
	return {
		key: frozenset(site.callee for site in group)
		for key, group in groupby(sorted(sites, key=_site_order), key=_site_key)
	}


def _site_key(site: Site) -> tuple[str, SourceLocation]:
	return (site.function, site.location)


def _site_order(site: Site) -> tuple[str, str, int, int]:
	return (site.function, site.location.file, site.location.line, site.location.column)


def _names_by_address(program: Program) -> Mapping[Address, frozenset[str]]:
	return {
		address: frozenset(name for _, name in group)
		for address, group in groupby(
			sorted(
				(address, name)
				for name, addresses in program.symbol_addresses.items()
				for address in addresses
			),
			key=_address_of,
		)
	}


def _address_of(entry: tuple[Address, str]) -> Address:
	return entry[0]


def _site_field(
	callees: Mapping[tuple[str, SourceLocation], frozenset[Descriptor]],
	names: frozenset[str],
	location: SourceLocation,
) -> Field | None:
	match tuple(
		frozenset(callee for name in names for callee in callees.get((name, location), frozenset()))
	):
		case (Field() as field,):
			return field
		case _:
			return None
