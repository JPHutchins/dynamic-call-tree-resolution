# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The state each successor of a block starts from, narrowed by the branch that leads to it."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from salix import replace

from dynamic_call_tree_resolution.vsa.abi import arm_predicated
from dynamic_call_tree_resolution.vsa.cfg import CallsNoReturn
from dynamic_call_tree_resolution.vsa.conditions import Conditional, edge_values

if TYPE_CHECKING:
	from collections.abc import Mapping

	from capstone import CsInsn

	from dynamic_call_tree_resolution.model import Address
	from dynamic_call_tree_resolution.vsa.cfg import Block
	from dynamic_call_tree_resolution.vsa.state import State


def successor_states(
	block: Block, before_last: State, outgoing: State
) -> tuple[tuple[Address, State], ...]:
	"""Each successor some value can reach, with the state its edge carries there.

	A predicated last instruction did not execute on the fallthrough edge, so that edge starts
	from the state before it.
	"""
	match block.condition:
		case None:
			return tuple((successor, outgoing) for successor in block.successors)
		case Conditional(test=test, taken=taken, fallthrough=fallthrough):
			edges = edge_values(test, before_last.registers)
			last = block.instructions[-1]
			return tuple(
				(successor, _narrowed(state, edges[outcome], before_last, last))
				for outcome, successors, state in (
					(False, fallthrough, before_last if arm_predicated(last) else outgoing),
					(True, taken, outgoing),
				)
				if outcome in edges
				for successor in successors
			)
		case CallsNoReturn():
			return ()
		case _ as unreachable:
			assert_never(unreachable)


def _narrowed(
	state: State, values: Mapping[int, frozenset[Address]], before_last: State, last: CsInsn
) -> State:
	"""The state with each tested register cut to its edge's values, where the test saw it."""
	written = frozenset[int]() if state is before_last else frozenset(last.regs_access()[1])
	narrowed = {
		register: held
		for register, held in values.items()
		if register not in written
		and held != state.registers.get(register)
		and state.registers.get(register) == before_last.registers.get(register)
	}
	return replace(state, registers={**state.registers, **narrowed}) if narrowed else state
