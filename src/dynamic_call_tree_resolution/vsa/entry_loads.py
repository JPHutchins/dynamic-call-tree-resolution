# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The words a function loads through its arguments before anything it runs could change them."""

from __future__ import annotations

from itertools import accumulate, takewhile
from typing import TYPE_CHECKING, Final, assert_never

from capstone import arm_const
from salix import Struct

from dynamic_call_tree_resolution.model import Address, InstructionFamily
from dynamic_call_tree_resolution.vsa.abi import (
	ARM_ARGUMENT_REGISTERS,
	ARM_DESCENDING_STORES,
	ARM_EXCLUSIVE_STORES,
	ARM_MULTIPLE_STORES,
	ARM_STORE_WIDTHS,
	ARM_UNTRACKED_STORES,
	arm_predicated,
	normalized,
)
from dynamic_call_tree_resolution.vsa.cfg import predecessors

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping

	from capstone import CsInsn

	from dynamic_call_tree_resolution.model import Program
	from dynamic_call_tree_resolution.vsa.cfg import FunctionGraph


class EntryLoad(Struct):
	"""A word load through an argument register that nothing before it in the function wrote."""

	site: Address
	register: int
	offset: int


def entry_loads(
	program: Program, graphs: Iterable[FunctionGraph]
) -> Mapping[Address, tuple[EntryLoad, ...]]:
	"""Each function's entry loads, by where its code starts."""
	match program.machine.family:
		case InstructionFamily.ARM:
			return {
				normalized(graph.function.address, program.machine): loads
				for graph in graphs
				if (loads := _arm_entry_loads(graph))
			}
		case InstructionFamily.X86:
			return {}
		case _ as unreachable:
			assert_never(unreachable)


def _arm_entry_loads(graph: FunctionGraph) -> tuple[EntryLoad, ...]:
	entry = graph.blocks[0]
	if entry.start in predecessors({block.start: block for block in graph.blocks}):
		return ()
	prefix = tuple(takewhile(_leaves_memory_alone, entry.instructions))
	return tuple(
		load
		for instruction, written in zip(
			prefix,
			accumulate(map(_written, prefix), _union, initial=frozenset[int]()),
			strict=False,
		)
		if (load := _entry_load(instruction, written)) is not None
	)


_CALLS: Final = frozenset({arm_const.ARM_INS_BL, arm_const.ARM_INS_BLX, arm_const.ARM_INS_SVC})
_STORES: Final = frozenset(
	{*ARM_STORE_WIDTHS, *ARM_MULTIPLE_STORES, *ARM_UNTRACKED_STORES, *ARM_EXCLUSIVE_STORES}
)


def _leaves_memory_alone(instruction: CsInsn) -> bool:
	return instruction.id not in _CALLS and (instruction.id not in _STORES or _pushes(instruction))


def _pushes(instruction: CsInsn) -> bool:
	return instruction.id in ARM_DESCENDING_STORES and (
		instruction.id in (arm_const.ARM_INS_PUSH, arm_const.ARM_INS_VPUSH)
		or (instruction.writeback and instruction.operands[0].reg == arm_const.ARM_REG_SP)
	)


def _written(instruction: CsInsn) -> frozenset[int]:
	return frozenset(instruction.regs_access()[1])


def _union(written: frozenset[int], more: frozenset[int]) -> frozenset[int]:
	return written | more


def _entry_load(instruction: CsInsn, written: frozenset[int]) -> EntryLoad | None:
	if (
		instruction.id != arm_const.ARM_INS_LDR
		or arm_predicated(instruction)
		or instruction.writeback
		or len(instruction.operands) != 2
	):
		return None
	memory = instruction.operands[1].mem
	return (
		EntryLoad(Address(instruction.address), memory.base, memory.disp)
		if memory.base in ARM_ARGUMENT_REGISTERS
		and memory.base not in written
		and memory.index == 0
		else None
	)
