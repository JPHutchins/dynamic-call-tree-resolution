# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The value-set lattice and its map encoding."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING, Final, assert_never

from salix import Struct

from dynamic_call_tree_resolution.model import Address

if TYPE_CHECKING:
	from collections.abc import Mapping


K_BOUND: Final = 64


class Top(Struct):
	"""The value set of anything."""


class Known[T: Hashable](Struct):
	"""A bounded value set."""

	values: frozenset[T]


type Lattice[T: Hashable] = Top | Known[T]


type ValueSet = Lattice[Address]


type OffsetSet = Lattice[int]


def join_sets[T: Hashable](current: frozenset[T], incoming: frozenset[T]) -> Lattice[T]:
	return capped(current | incoming)


def join_maps[K: int, T: Hashable](
	current: Mapping[K, frozenset[T]], incoming: Mapping[K, frozenset[T]]
) -> dict[K, frozenset[T]]:
	return {
		key: joined
		for key in current.keys() & incoming.keys()
		if len(joined := current[key] | incoming[key]) <= K_BOUND
	}


def lookup[K: int, T: Hashable](mapping: Mapping[K, frozenset[T]], key: K) -> Lattice[T]:
	return Known(values=mapping[key]) if key in mapping else Top()


def map_entry[K: int, T: Hashable](key: K, value: Lattice[T]) -> tuple[tuple[K, frozenset[T]], ...]:
	match value:
		case Top():
			return ()
		case Known(values=values):
			return ((key, values),)
		case _ as unreachable:
			assert_never(unreachable)


def capped[T: Hashable](values: frozenset[T]) -> Lattice[T]:
	return Top() if len(values) > K_BOUND else Known(values=values)


def bind[T: Hashable, U: Hashable](
	value: Lattice[T], function: Callable[[frozenset[T]], Lattice[U]]
) -> Lattice[U]:
	match value:
		case Top():
			return Top()
		case Known(values=values):
			return function(values)
		case _ as unreachable:
			assert_never(unreachable)


def put_value[K: int, T: Hashable](
	mapping: Mapping[K, frozenset[T]], key: K, value: Lattice[T]
) -> dict[K, frozenset[T]]:
	copied = dict(mapping)
	copied.pop(key, None)
	copied.update(map_entry(key, value))
	return copied


def map_set[T: Hashable, U: Hashable](values: Lattice[T], function: Callable[[T], U]) -> Lattice[U]:
	return bind(values, lambda known: Known(values=frozenset(function(value) for value in known)))


def shift_addresses(values: ValueSet, delta: int) -> ValueSet:
	return map_set(values, lambda address: Address(address + delta))


def shift_offsets(values: OffsetSet, delta: int) -> OffsetSet:
	return map_set(values, lambda offset: offset + delta)


def indexed(base: ValueSet, index: ValueSet, scale: int, disp: int) -> ValueSet:
	return bind(
		base,
		lambda bases: bind(
			index,
			lambda indexes: capped(
				frozenset(Address(b + i * scale + disp) for b in bases for i in indexes)
			),
		),
	)


def indexed_offsets(
	base: frozenset[int], index: frozenset[Address], scale: int, disp: int
) -> OffsetSet:
	return capped(frozenset(b + i * scale + disp for b in base for i in index))
