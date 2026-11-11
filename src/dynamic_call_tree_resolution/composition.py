# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Function composition the standard library leaves out."""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING

from salix import Struct

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Iterator


def flatmap[T, U](function: Callable[[T], Iterable[U]], items: Iterable[T]) -> Iterator[U]:
	"""Each item's results, one item after another.

	>>> tuple(flatmap(range, (1, 2, 3)))
	(0, 0, 1, 0, 1, 2)
	>>> tuple(flatmap(lambda word: word.split("-"), ("a-b", "c")))
	('a', 'b', 'c')
	"""
	return chain.from_iterable(map(function, items))


class Pipeline[T](Struct):
	"""A value, and the functions applied to it in reading order.

	>>> Pipeline((3, 1, 2)).then(sorted).then(tuple).value
	(1, 2, 3)
	>>> from functools import partial
	>>> Pipeline(("ab", "cd")).then(partial(flatmap, reversed)).then("".join).value
	'badc'
	"""

	value: T

	def then[U](self, function: Callable[[T], U]) -> Pipeline[U]:
		return Pipeline(function(self.value))
