# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Addresses a register may hold, and how they join."""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Final

from salix import Struct

if TYPE_CHECKING:
	from collections.abc import Callable, Mapping


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


def join(
	current: Registers | None, incoming: Registers, *, carries: Callable[[int], bool] | None
) -> tuple[Registers, Held]:
	"""The registers where control flow meets, and what escapes there.

	``carries`` is ``None`` along a forward edge. Along an edge back into a loop, it says whether
	the loop carries what the instruction at an address materialized.
	"""
	if current is None:
		held, overflow = _capped(incoming.held)
		return Registers(held=held, constants=incoming.constants), overflow
	joined = {
		register: _joined(
			current.held.get(register, NOTHING), incoming.held.get(register, NOTHING), carries
		)
		for register in current.held.keys() | incoming.held.keys()
	}
	held, overflow = _capped({register: kept for register, (kept, _) in joined.items()})
	return Registers(
		held=held,
		constants={
			register: value
			for register, value in current.constants.items()
			if incoming.constants.get(register) == value
		},
	), union(overflow, *(escaped for _, escaped in joined.values()))


def _joined(
	current: Held, incoming: Held, carries: Callable[[int], bool] | None
) -> tuple[Held, Held]:
	"""What one register holds after a join, and what escapes there.

	An address a loop carries stays in the object it entered the loop pointing into (C's pointer
	provenance), so it becomes that address plus an unknown offset. A new address the loop can't
	be shown to carry escapes instead, which keeps the join from growing.
	"""
	if carries is None:
		return union(current, incoming), NOTHING
	arrived = frozenset(
		derived
		for derived in chain(incoming.exact, incoming.indexed)
		if derived not in current.exact and derived not in current.indexed
	)
	moved = frozenset(map(_lineage, arrived)) & frozenset(
		map(_lineage, chain(current.exact, current.indexed))
	)
	carried = frozenset(lineage for lineage in moved if carries(lineage[1]))
	return Held(
		exact=frozenset(
			derived
			for derived in current.exact | incoming.exact
			if _lineage(derived) not in moved
			or (derived in current.exact and _lineage(derived) not in carried)
		),
		indexed=frozenset(
			derived
			for derived in chain(current.indexed, incoming.indexed)
			if _lineage(derived) not in moved
			or (derived in current.indexed and _lineage(derived) not in carried)
		)
		| frozenset(
			derived
			for derived in chain(current.exact, current.indexed)
			if _lineage(derived) in carried
		),
	), Held(
		exact=frozenset(
			derived for derived in arrived & incoming.exact if _lineage(derived) in moved - carried
		),
		indexed=frozenset(
			derived
			for derived in arrived & incoming.indexed
			if _lineage(derived) in moved - carried
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
