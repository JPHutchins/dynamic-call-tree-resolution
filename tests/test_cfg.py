# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The block structure of a function's control-flow graph."""

from __future__ import annotations

from dynamic_call_tree_resolution.model import Address
from dynamic_call_tree_resolution.vsa.cfg import Block, predecessors


def _block(start: int, *successors: int) -> Block:
	return Block(
		start=Address(start),
		instructions=(),
		successors=tuple(map(Address, successors)),
		condition=None,
	)


def test_a_block_keeps_every_predecessor_when_another_edge_sorts_between_them() -> None:
	assert predecessors(
		{
			block.start: block
			for block in (_block(0x10, 0x30), _block(0x20, 0x40), _block(0x28, 0x30))
		}
	) == {Address(0x30): (Address(0x10), Address(0x28)), Address(0x40): (Address(0x20),)}
