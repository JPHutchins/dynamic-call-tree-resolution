# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The state each successor of a block starts from, narrowed by the branch that leads to it."""

from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, assert_never

from salix import replace

from dynamic_call_tree_resolution.vsa.abi import arm_predicated
from dynamic_call_tree_resolution.vsa.cfg import CallsNoReturn
from dynamic_call_tree_resolution.vsa.conditions import Conditional, edge_values

if TYPE_CHECKING:
	from collections.abc import Callable, Mapping

	from capstone import CsInsn

	from dynamic_call_tree_resolution.model import Address
	from dynamic_call_tree_resolution.vsa.cfg import Block
	from dynamic_call_tree_resolution.vsa.cfg.narrowing import Narrowing
	from dynamic_call_tree_resolution.vsa.state import State


def successor_states(
	block: Block, states: tuple[State, ...], step: Callable[[State, CsInsn], State]
) -> tuple[tuple[Address, State], ...]:
	"""Each successor some value can reach, with the state its edge carries there.

	The states are those before each instruction and after the last. A predicated last
	instruction did not execute on the fallthrough edge, so that edge ends before it.
	"""
	match block.condition:
		case None:
			return tuple((successor, states[-1]) for successor in block.successors)
		case Conditional(test=test, taken=taken, fallthrough=fallthrough, narrowing=narrowing):
			edges = edge_values(test, states[narrowing.at].registers)
			return tuple(
				(successor, _edge_state(block, states, step, narrowing, edges[outcome], end))
				for outcome, successors, end in (
					(
						False,
						fallthrough,
						len(states) - (2 if arm_predicated(block.instructions[-1]) else 1),
					),
					(True, taken, len(states) - 1),
				)
				if outcome in edges
				for successor in successors
			)
		case CallsNoReturn():
			return ()
		case _ as unreachable:
			assert_never(unreachable)


def _edge_state(
	block: Block,
	states: tuple[State, ...],
	step: Callable[[State, CsInsn], State],
	narrowing: Narrowing,
	values: Mapping[int, frozenset[Address]],
	end: int,
) -> State:
	"""The block replayed to the edge from where the test narrows, when that narrows anything."""
	narrowed = _narrowed(states[narrowing.at], values, narrowing.copies)
	return (
		states[end]
		if narrowed is states[narrowing.at]
		else reduce(step, block.instructions[narrowing.at : end], narrowed)
	)


def _narrowed(
	state: State, values: Mapping[int, frozenset[Address]], copies: tuple[tuple[int, int], ...]
) -> State:
	narrowed = {
		register: held
		for register, held in (
			*values.items(),
			*((source, values[copy]) for copy, source in copies if copy in values),
		)
		if held != state.registers.get(register)
	}
	return replace(state, registers={**state.registers, **narrowed}) if narrowed else state
