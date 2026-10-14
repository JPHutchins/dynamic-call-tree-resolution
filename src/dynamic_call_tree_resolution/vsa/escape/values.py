# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Addresses a register may hold, and how they join."""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Final

from salix import Struct

if TYPE_CHECKING:
	from collections.abc import Mapping


_TRACKED_LIMIT: Final = 64


WORD_MASK: Final = 0xFFFF_FFFF


class Derived(Struct):
	"""An address the code computed from one it materialized."""

	origin: int
	address: int
	site: int
	"""The instruction that materialized the origin."""


class Held(Struct):
	"""The addresses one register may hold."""

	exact: frozenset[Derived] = frozenset()
	indexed: frozenset[Derived] = frozenset()
	"""Addresses with an offset the scan doesn't know added."""


NOTHING: Final = Held()


class Registers(Struct):
	"""What the registers may hold on entry to an instruction."""

	held: Mapping[int, Held]
	constants: Mapping[int, int]
	"""What every path into the instruction leaves in the register."""


UNKNOWN: Final = Registers(held={}, constants={})


def join(current: Registers | None, incoming: Registers, *, back: bool) -> tuple[Registers, Held]:
	held, overflow = _capped(
		incoming.held
		if current is None
		else {
			register: (_carried if back else union)(
				current.held.get(register, NOTHING), incoming.held.get(register, NOTHING)
			)
			for register in current.held.keys() | incoming.held.keys()
		}
	)
	return Registers(
		held=held,
		constants=incoming.constants
		if current is None
		else {
			register: value
			for register, value in current.constants.items()
			if incoming.constants.get(register) == value
		},
	), overflow


def _carried(current: Held, incoming: Held) -> Held:
	"""The join along an edge back into a loop.

	An address the loop moves stays in the object it entered the loop pointing into (C's pointer
	provenance), so it becomes that address plus an unknown offset.
	"""
	moved = frozenset(
		_lineage(derived)
		for derived in chain(incoming.exact, incoming.indexed)
		if derived not in current.exact and derived not in current.indexed
	) & frozenset(_lineage(derived) for derived in chain(current.exact, current.indexed))
	return Held(
		exact=frozenset(
			derived for derived in current.exact | incoming.exact if _lineage(derived) not in moved
		),
		indexed=frozenset(
			derived
			for derived in chain(current.indexed, incoming.indexed)
			if _lineage(derived) not in moved
		)
		| frozenset(
			derived
			for derived in chain(current.exact, current.indexed)
			if _lineage(derived) in moved
		),
	)


def _lineage(derived: Derived) -> tuple[int, int]:
	return derived.origin, derived.site


def _capped(held: Mapping[int, Held]) -> tuple[Mapping[int, Held], Held]:
	"""The registers that hold few enough addresses to follow, and what the rest held."""
	return (
		{register: value for register, value in held.items() if 0 < size(value) <= _TRACKED_LIMIT},
		union(*(value for value in held.values() if size(value) > _TRACKED_LIMIT)),
	)


def size(held: Held) -> int:
	return len(held.exact) + len(held.indexed)


def union(*held: Held) -> Held:
	match tuple(value for value in held if value.exact or value.indexed):
		case ():
			return NOTHING
		case (only,):
			return only
		case several:
			return Held(
				exact=frozenset[Derived]().union(*(value.exact for value in several)),
				indexed=frozenset[Derived]().union(*(value.indexed for value in several)),
			)


def offset_by(held: Held, delta: int) -> Held:
	return Held(
		exact=frozenset(_moved_by(derived, delta) for derived in held.exact),
		indexed=frozenset(_moved_by(derived, delta) for derived in held.indexed),
	)


def _moved_by(derived: Derived, delta: int) -> Derived:
	return Derived(
		origin=derived.origin, address=(derived.address + delta) & WORD_MASK, site=derived.site
	)


def unknown_offset(held: Held) -> Held:
	return Held(indexed=held.exact | held.indexed)
