# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Transfer functions for x86 instructions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from capstone import (
	x86_const,
)

from dynamic_call_tree_resolution.model import Address, Machine
from dynamic_call_tree_resolution.vsa.abi import SP_REGISTERS, X86_MOVES
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
	load_value,
	memory_addresses,
	stack_offsets,
	store_value,
)
from dynamic_call_tree_resolution.vsa.state import (
	State,
	copy_register,
	set_offsets,
	set_register,
	stack_read,
	top_written,
	union_write,
)

if TYPE_CHECKING:
	from capstone import CsInsn


def apply_x86(context: Context, instruction: CsInsn, state: State) -> State:
	machine = context.program.machine
	if instruction.mnemonic in X86_MOVES:
		destination, source = instruction.operands
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
					context, state, instruction, destination, lookup(state.registers, source.reg)
				)
			case (x86_const.X86_OP_MEM, x86_const.X86_OP_IMM):
				return store_value(
					context,
					state,
					instruction,
					destination,
					Known(values=frozenset({Address(source.imm)})),
				)
			case _:
				return top_written(instruction, state)  # pragma: no cover
	if instruction.mnemonic == "lea":
		destination, source = instruction.operands
		memory = source.mem
		if memory.base == x86_const.X86_REG_RIP:
			address = Address(instruction.address + instruction.size + memory.disp)
			return set_register(state, destination.reg, Known(values=frozenset({address})))
		if memory.base in SP_REGISTERS[machine]:
			return set_offsets(state, destination.reg, stack_offsets(state, memory, memory.base))
		return set_register(state, destination.reg, memory_addresses(context, state, memory))
	if instruction.mnemonic in ("inc", "dec"):
		destination = instruction.operands[0]
		if destination.type != x86_const.X86_OP_REG:  # pragma: no branch
			return top_written(instruction, state)
		return _shift_x86_destination(
			state,
			machine,
			destination.reg,
			destination.reg,
			1 if instruction.mnemonic == "inc" else -1,
		)
	if instruction.mnemonic in ("add", "sub"):
		destination, source = instruction.operands
		if destination.type != x86_const.X86_OP_REG or source.type != x86_const.X86_OP_IMM:
			return top_written(instruction, state)
		delta = source.imm if instruction.mnemonic == "add" else -source.imm
		return _shift_x86_destination(state, machine, destination.reg, destination.reg, delta)
	if instruction.mnemonic == "xor":
		destination, source = instruction.operands
		if (
			destination.type == x86_const.X86_OP_REG
			and source.type == x86_const.X86_OP_REG
			and destination.reg == source.reg
		):
			return set_register(state, destination.reg, Known(values=frozenset({Address(0)})))
		return top_written(instruction, state)
	if instruction.mnemonic == "push":
		operand = instruction.operands[0]
		value = (
			Known(values=frozenset({Address(operand.imm)}))
			if operand.type == x86_const.X86_OP_IMM
			else lookup(state.registers, operand.reg)
			if operand.type == x86_const.X86_OP_REG
			else load_value(context, state, instruction, operand)
		)
		return _push(state, SP_REGISTERS[machine][0], context.program.pointer_size, value)
	if instruction.mnemonic == "pop":
		operand = instruction.operands[0]
		return _pop(state, SP_REGISTERS[machine][0], context.program.pointer_size, operand.reg)
	return top_written(instruction, state)


def _shift_x86_destination(
	state: State, machine: Machine, destination: int, source: int, delta: int
) -> State:
	if destination in SP_REGISTERS[machine]:
		return set_offsets(
			state, destination, shift_offsets(lookup(state.sp_offsets, destination), delta)
		)
	return set_register(state, destination, shift_addresses(lookup(state.registers, source), delta))


def _push(state: State, sp: int, pointer_size: int, value: ValueSet) -> State:
	offsets = shift_offsets(lookup(state.sp_offsets, sp), -pointer_size)
	return State(
		registers=state.registers,
		sp_offsets=put_value(state.sp_offsets, sp, offsets),
		stack=union_write(state.stack, offsets, value),
		globals=state.globals,
	)


def _pop(state: State, sp: int, pointer_size: int, destination: int) -> State:
	offsets = lookup(state.sp_offsets, sp)
	return State(
		registers=put_value(state.registers, destination, stack_read(state.stack, offsets)),
		sp_offsets=put_value(state.sp_offsets, sp, shift_offsets(offsets, pointer_size)),
		stack=state.stack,
		globals=state.globals,
	)
