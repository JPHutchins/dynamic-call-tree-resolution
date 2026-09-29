# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Transfer functions for Thumb instructions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from capstone import arm_const

from dynamic_call_tree_resolution.model import Address
from dynamic_call_tree_resolution.vsa.abi import ARM_LOADS, ARM_MOVES
from dynamic_call_tree_resolution.vsa.lattice import (
	Known,
	lookup,
	map_set,
	put_value,
	shift_addresses,
	shift_offsets,
)
from dynamic_call_tree_resolution.vsa.memory import (
	Context,
	load_value,
	scaled_addresses,
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
	from capstone import ArmCsOperand, CsInsn, CsOperand


def _arm_operands(instruction: CsInsn) -> tuple[ArmCsOperand, ...]:
	return tuple(cast("ArmCsOperand", operand) for operand in instruction.operands)


def apply_arm(context: Context, instruction: CsInsn, state: State) -> State:
	base_mnemonic = instruction.mnemonic.split(".")[0]
	if base_mnemonic == "push":
		return _arm_push(instruction, state)
	if base_mnemonic == "pop":
		return _arm_pop(instruction, state)
	if base_mnemonic in ("movw", "movt"):
		destination, source = _arm_operands(instruction)
		if source.type != arm_const.ARM_OP_IMM:  # pragma: no branch
			return top_written(instruction, state)  # pragma: no cover
		if base_mnemonic == "movw":
			return set_register(
				state, destination.reg, Known(values=frozenset({Address(source.imm)}))
			)
		return set_register(
			state,
			destination.reg,
			map_set(
				lookup(state.registers, destination.reg),
				lambda low: Address((source.imm << 16) | (low & 0xFFFF)),
			),
		)
	if instruction.mnemonic in ARM_LOADS:
		load_destination, load_source = _arm_operands(instruction)[0], instruction.operands[1]
		value = load_value(context, state, instruction, load_source)
		if len(instruction.operands) == 3:
			offset = instruction.operands[2]
			updated = _advance_post_index(context, state, load_source, offset.imm)
			return set_register(updated, load_destination.reg, value)
		return set_register(state, load_destination.reg, value)
	if instruction.mnemonic in ("str", "str.w"):
		store_source, store_destination = _arm_operands(instruction)[0], instruction.operands[1]
		if store_source.type != arm_const.ARM_OP_REG:  # pragma: no branch
			return top_written(instruction, state)  # pragma: no cover
		stored = store_value(
			context,
			state,
			instruction,
			store_destination,
			lookup(state.registers, store_source.reg),
		)
		if len(instruction.operands) == 3:
			return _advance_post_index(
				context, stored, store_destination, instruction.operands[2].imm
			)
		return stored
	if instruction.mnemonic in ARM_MOVES:
		destination, source = _arm_operands(instruction)
		match source.type:
			case arm_const.ARM_OP_REG:
				return copy_register(state, destination.reg, source.reg)
			case arm_const.ARM_OP_IMM:
				return set_register(
					state, destination.reg, Known(values=frozenset({Address(source.imm)}))
				)
			case _:
				return top_written(instruction, state)  # pragma: no cover
	if base_mnemonic in ("add", "adds", "sub", "subs"):
		return _arm_arithmetic(context, instruction, state, base_mnemonic)
	return top_written(instruction, state)


def _advance_post_index(context: Context, state: State, operand: CsOperand, delta: int) -> State:
	base = operand.mem.base
	if base == arm_const.ARM_REG_SP:
		return set_offsets(state, base, shift_offsets(lookup(state.sp_offsets, base), delta))
	return set_register(state, base, shift_addresses(lookup(state.registers, base), delta))


def _arm_shift_scale(operand: ArmCsOperand) -> int | None:
	if operand.shift.type == arm_const.ARM_SFT_LSL:
		return 1 << operand.shift.value
	if operand.shift.type == arm_const.ARM_SFT_INVALID:
		return 1
	return None


def _arm_arithmetic(
	context: Context, instruction: CsInsn, state: State, base_mnemonic: str
) -> State:
	operands = _arm_operands(instruction)
	destination = operands[0]
	sign = 1 if base_mnemonic.startswith("add") else -1
	if len(operands) == 2:
		source = operands[1]
		if source.type != arm_const.ARM_OP_IMM:  # pragma: no branch
			return top_written(instruction, state)  # pragma: no cover
		return _arm_shift_destination(state, destination.reg, destination.reg, sign * source.imm)
	register_operand, value_operand = operands[1], operands[2]
	if value_operand.type == arm_const.ARM_OP_IMM:
		return _arm_shift_destination(
			state, destination.reg, register_operand.reg, sign * value_operand.imm
		)
	if value_operand.type == arm_const.ARM_OP_REG:
		scale = _arm_shift_scale(value_operand)
		if scale is None:
			return top_written(instruction, state)
		return set_register(
			state,
			destination.reg,
			scaled_addresses(
				context,
				lookup(state.registers, register_operand.reg),
				lookup(state.registers, value_operand.reg),
				scale,
				0,
			),
		)
	return top_written(instruction, state)  # pragma: no cover


def _arm_shift_destination(state: State, destination: int, source: int, delta: int) -> State:
	if destination == arm_const.ARM_REG_SP:
		return set_offsets(
			state, destination, shift_offsets(lookup(state.sp_offsets, destination), delta)
		)
	if source == arm_const.ARM_REG_SP:
		return set_offsets(
			state, destination, shift_offsets(lookup(state.sp_offsets, source), delta)
		)
	return set_register(state, destination, shift_addresses(lookup(state.registers, source), delta))


def _arm_push(instruction: CsInsn, state: State) -> State:
	operands = _arm_operands(instruction)
	registers = tuple(operand.reg for operand in operands if operand.reg != arm_const.ARM_REG_SP)
	offsets = shift_offsets(lookup(state.sp_offsets, arm_const.ARM_REG_SP), -4 * len(registers))
	stack = state.stack
	for index, register in enumerate(registers):
		stack = union_write(
			stack, shift_offsets(offsets, 4 * index), lookup(state.registers, register)
		)
	return State(
		registers=state.registers,
		sp_offsets=put_value(state.sp_offsets, arm_const.ARM_REG_SP, offsets),
		stack=stack,
		globals=state.globals,
	)


def _arm_pop(instruction: CsInsn, state: State) -> State:
	operands = _arm_operands(instruction)
	registers = tuple(
		operand.reg
		for operand in operands
		if operand.reg not in (arm_const.ARM_REG_SP, arm_const.ARM_REG_PC)
	)
	offsets = lookup(state.sp_offsets, arm_const.ARM_REG_SP)
	registers_map = state.registers
	for index, register in enumerate(registers):
		registers_map = put_value(
			registers_map, register, stack_read(state.stack, shift_offsets(offsets, 4 * index))
		)
	return State(
		registers=registers_map,
		sp_offsets=put_value(
			state.sp_offsets, arm_const.ARM_REG_SP, shift_offsets(offsets, 4 * len(operands))
		),
		stack=state.stack,
		globals=state.globals,
	)
