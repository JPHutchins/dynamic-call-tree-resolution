# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""A half-open range of addresses."""

from __future__ import annotations

from typing import TYPE_CHECKING

from salix import Struct

if TYPE_CHECKING:
	from dynamic_call_tree_resolution.model import Address


class Span(Struct, order=True):
	"""The addresses from start up to, not including, end; spans order by start, then end.

	>>> from dynamic_call_tree_resolution.model import Address
	>>> sorted((Span(start=Address(2), end=Address(3)), Span(start=Address(1), end=Address(5))))
	[Span(start=1, end=5), Span(start=2, end=3)]
	>>> Address(4) in Span(start=Address(1), end=Address(5)), Address(5) in Span(start=Address(1), end=Address(5))
	(True, False)
	"""

	start: Address
	end: Address

	def __contains__(self, address: int) -> bool:
		return self.start <= address < self.end
