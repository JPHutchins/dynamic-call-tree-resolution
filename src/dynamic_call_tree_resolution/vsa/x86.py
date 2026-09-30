# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Transfer functions for x86 instructions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from capstone import x86_const

from dynamic_call_tree_resolution.model import Address, Machine
from dynamic_call_tree_resolution.vsa.abi import (
	SP_REGISTERS,
	X86_MEMORY_READERS,
	X86_MOVES,
	X86_REPEATS,
	X86_UNBOUNDED_STORES,
)
from dynamic_call_tree_resolution.vsa.lattice import (
	Known,
	ValueSet,
	lookup,
	put_value,
	shift_addresses,
	shift_offsets,
)
from dynamic_call_tree_resolution.vsa.memory import (
	Context,
	Frame,
	anywhere,
	load_value,
	memory_addresses,
	memory_destination,
	stack_offsets,
	store_at,
	store_value,
)
from dynamic_call_tree_resolution.vsa.state import (
	State,
	Store,
	copy_register,
	frame_based,
	set_offsets,
	set_register,
	stack_read,
	top_written,
)

if TYPE_CHECKING:
	from capstone import CsInsn, X86CsOperand


def _x86_operands(instruction: CsInsn) -> tuple[X86CsOperand, ...]:
	return tuple(cast("X86CsOperand", operand) for operand in instruction.operands)


def apply_x86(context: Context, instruction: CsInsn, state: State) -> State:
	machine = context.program.machine
	if instruction.mnemonic in X86_MOVES:
		destination, source = _x86_operands(instruction)
		match (destination.type, source.type):
			case (x86_const.X86_OP_REG, x86_const.X86_OP_REG):
				return copy_register(state, destination.reg, source.reg)
			case (x86_const.X86_OP_REG, x86_const.X86_OP_IMM):
				return set_register(
					state, destination.reg, Known(values=frozenset({Address(source.imm)}))
				)
			case (x86_const.X86_OP_REG, x86_const.X86_OP_MEM):
				return set_register(
					state, destination.reg, load_value(context, state, instruction, source)
				)
			case (x86_const.X86_OP_MEM, x86_const.X86_OP_REG):
				return store_value(
					context,
					state,
					instruction,
					destination,
					Store(words=(lookup(state.registers, source.reg),), width=destination.size),
				)
			case (x86_const.X86_OP_MEM, x86_const.X86_OP_IMM):
				return store_value(
					context,
					state,
					instruction,
					destination,
					Store(
						words=(Known(values=frozenset({Address(source.imm)})),),
						width=destination.size,
					),
				)
			case _:
				return top_written(instruction, state)  # pragma: no cover
	if instruction.mnemonic == "lea":
		destination, source = _x86_operands(instruction)
		memory = source.mem
		if memory.base == x86_const.X86_REG_RIP:
			address = Address(instruction.address + instruction.size + memory.disp)
			return set_register(state, destination.reg, Known(values=frozenset({address})))
		if frame_based(state, machine, memory.base):
			return set_offsets(state, destination.reg, stack_offsets(state, memory, memory.base))
		return set_register(state, destination.reg, memory_addresses(context, state, memory))
	if instruction.mnemonic in ("inc", "dec"):
		destination = _x86_operands(instruction)[0]
		if destination.type != x86_const.X86_OP_REG:
			return _clobbered(context, instruction, state)
		return _shift_x86_destination(
			state,
			machine,
			destination.reg,
			destination.reg,
			1 if instruction.mnemonic == "inc" else -1,
		)
	if instruction.mnemonic in ("add", "sub"):
		destination, source = _x86_operands(instruction)
		if destination.type != x86_const.X86_OP_REG or source.type != x86_const.X86_OP_IMM:
			return _clobbered(context, instruction, state)
		delta = source.imm if instruction.mnemonic == "add" else -source.imm
		return _shift_x86_destination(state, machine, destination.reg, destination.reg, delta)
	if instruction.mnemonic == "xor":
		destination, source = _x86_operands(instruction)
		if (
			destination.type == x86_const.X86_OP_REG
			and source.type == x86_const.X86_OP_REG
			and destination.reg == source.reg
		):
			return set_register(state, destination.reg, Known(values=frozenset({Address(0)})))
		return _clobbered(context, instruction, state)
	if instruction.mnemonic == "push":
		operand = _x86_operands(instruction)[0]
		value = (
			Known(values=frozenset({Address(operand.imm)}))
			if operand.type == x86_const.X86_OP_IMM
			else lookup(state.registers, operand.reg)
			if operand.type == x86_const.X86_OP_REG
			else load_value(context, state, instruction, operand)
		)
		return _push(context, state, SP_REGISTERS[machine][0], value)
	if instruction.mnemonic == "pop":
		operand = _x86_operands(instruction)[0]
		if operand.type != x86_const.X86_OP_REG:
			return _clobbered(context, instruction, state)
		return _pop(state, SP_REGISTERS[machine][0], context.program.pointer_size, operand.reg)
	return _clobbered(context, instruction, state)


def _clobbered(context: Context, instruction: CsInsn, state: State) -> State:
	prefixes_and_name = instruction.mnemonic.split()
	operands = _x86_operands(instruction)
	if (
		not operands
		or operands[0].type != x86_const.X86_OP_MEM
		or prefixes_and_name[-1] in X86_MEMORY_READERS
	):
		return top_written(instruction, state)
	destination = memory_destination(context, state, instruction, operands[0].mem)
	return top_written(
		instruction,
		store_at(
			context,
			state,
			anywhere(destination)
			if prefixes_and_name[0] in X86_REPEATS or prefixes_and_name[-1] in X86_UNBOUNDED_STORES
			else destination,
			Store(words=(), width=operands[0].size),
		),
	)


def _shift_x86_destination(
	state: State, machine: Machine, destination: int, source: int, delta: int
) -> State:
	if frame_based(state, machine, destination):
		return set_offsets(
			state, destination, shift_offsets(lookup(state.sp_offsets, destination), delta)
		)
	return set_register(state, destination, shift_addresses(lookup(state.registers, source), delta))


def _push(context: Context, state: State, sp: int, value: ValueSet) -> State:
	pointer_size = context.program.pointer_size
	offsets = shift_offsets(lookup(state.sp_offsets, sp), -pointer_size)
	return set_offsets(
		store_at(context, state, Frame(offsets=offsets), Store(words=(value,), width=pointer_size)),
		sp,
		offsets,
	)


def _pop(state: State, sp: int, pointer_size: int, destination: int) -> State:
	offsets = lookup(state.sp_offsets, sp)
	return State(
		registers=put_value(state.registers, destination, stack_read(state.stack, offsets)),
		sp_offsets=put_value(state.sp_offsets, sp, shift_offsets(offsets, pointer_size)),
		stack=state.stack,
		globals=state.globals,
	)
