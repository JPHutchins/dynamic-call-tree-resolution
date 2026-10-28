# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The addresses a memory operand reaches, and what a load reads there."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from capstone import arm_const, x86_const
from salix import Struct

from dynamic_call_tree_resolution.model import Address, InstructionFamily
from dynamic_call_tree_resolution.points_to import instruction_set_at
from dynamic_call_tree_resolution.vsa.abi import (
	index_scale,
	program_counter,
)
from dynamic_call_tree_resolution.vsa.lattice import (
	Known,
	OffsetSet,
	Top,
	ValueSet,
	bind,
	indexed,
	indexed_offsets,
	lookup,
	shift_addresses,
	shift_offsets,
)
from dynamic_call_tree_resolution.vsa.memory.reads import (
	array_slots,
	image_value,
	low_bytes,
	section_slots,
)
from dynamic_call_tree_resolution.vsa.state import (
	State,
	Strided,
	frame_based,
	stack_read,
)

if TYPE_CHECKING:
	from capstone import CsInsn, CsMemOperand, CsOperand

	from dynamic_call_tree_resolution.model import Machine
	from dynamic_call_tree_resolution.vsa.memory.context import Context


def stack_offsets(context: Context, state: State, operand: CsOperand) -> OffsetSet:
	memory = operand.mem
	offsets = lookup(state.sp_offsets, memory.base)
	if memory.index == 0:
		return shift_offsets(offsets, memory.disp)
	match index_scale(context.program.machine, operand):
		case None:
			return Top()
		case int() as scale:
			return bind(
				offsets,
				lambda bases: bind(
					lookup(state.registers, memory.index),
					lambda indexes: indexed_offsets(bases, indexes, scale, memory.disp),
				),
			)
		case _ as unreachable:
			assert_never(unreachable)


def scaled_addresses(
	context: Context, base: ValueSet, index: ValueSet, scale: int, disp: int
) -> ValueSet:
	match index:
		case Top():
			return bind(
				base,
				lambda bases: (
					section_slots(context, Address(next(iter(bases)) + disp), scale)
					if len(bases) == 1
					else Top()
				),
			)
		case Known():
			return indexed(base, index, scale, disp)
		case _ as unreachable:
			assert_never(unreachable)


def memory_addresses(context: Context, state: State, operand: CsOperand) -> ValueSet:
	memory = operand.mem
	base = (
		Known(values=frozenset({Address(0)}))
		if memory.base == 0
		else lookup(state.registers, memory.base)
	)
	if memory.index == 0:
		return _displaced(context, state.strided.get(memory.base), base, memory.disp)
	match index_scale(context.program.machine, operand):
		case None:
			return Top()
		case int() as scale:
			return scaled_addresses(
				context, base, lookup(state.registers, memory.index), scale, memory.disp
			)
		case _ as unreachable:
			assert_never(unreachable)


def _displaced(context: Context, strided: Strided | None, base: ValueSet, disp: int) -> ValueSet:
	match strided:
		case None:
			return shift_addresses(base, disp)
		case Strided(base=array, stride=stride, offset=offset):
			return array_slots(context, array, Address(array + offset + disp), stride)
		case _ as unreachable:
			assert_never(unreachable)


class Loaded(Struct):
	"""A load's value, zero-extended from its width, and the one image address it reads."""

	value: ValueSet
	origin: Address | None


def load_value(context: Context, state: State, instruction: CsInsn, operand: CsOperand) -> ValueSet:
	return loaded(context, state, instruction, operand, context.program.pointer_size).value


def loaded(
	context: Context, state: State, instruction: CsInsn, operand: CsOperand, width: int
) -> Loaded:
	if (word := context.entry_words.get(Address(instruction.address))) is not None:
		return Loaded(word, None)
	match _load_addresses(context, state, instruction, operand):
		case None:
			return Loaded(
				low_bytes(
					context, stack_read(state.stack, stack_offsets(context, state, operand)), width
				),
				None,
			)
		case Known(values=values) as addresses if len(values) == 1:
			return Loaded(image_value(context, state, addresses, width), next(iter(values)))
		case Known() | Top() as addresses:
			return Loaded(
				image_value(context, state, addresses, width)
				if width >= context.program.pointer_size
				else Top(),
				None,
			)
		case _ as unreachable:
			assert_never(unreachable)


def _load_addresses(
	context: Context, state: State, instruction: CsInsn, operand: CsOperand
) -> ValueSet | None:
	memory = operand.mem
	if _pc_relative(context.program.machine, memory):
		return Known(
			values=frozenset(
				{
					Address(
						program_counter(
							instruction,
							instruction_set_at(context.program, Address(instruction.address)),
						)
						+ memory.disp
					)
				}
			)
		)
	if frame_based(state, context.program.machine, memory.base):
		return None
	return memory_addresses(context, state, operand)


def _pc_relative(machine: Machine, memory: CsMemOperand) -> bool:
	match machine.family:
		case InstructionFamily.X86:
			return memory.base == x86_const.X86_REG_RIP
		case InstructionFamily.ARM:
			return memory.base == arm_const.ARM_REG_PC and memory.index == 0
		case _ as unreachable:
			assert_never(unreachable)
