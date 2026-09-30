# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Transfer functions for Thumb instructions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from capstone import arm_const

from dynamic_call_tree_resolution.model import Address
from dynamic_call_tree_resolution.vsa.abi import (
	ARM_DESCENDING_STORES,
	ARM_EXCLUSIVE_STORES,
	ARM_LOADS,
	ARM_MOVES,
	ARM_MULTIPLE_STORES,
	ARM_REGISTER_BYTES,
	ARM_STORE_WIDTHS,
	ARM_STORED_REGISTERS,
	arm_mnemonic,
	arm_predicated,
)
from dynamic_call_tree_resolution.vsa.lattice import (
	Known,
	Top,
	lookup,
	map_set,
	put_value,
	shift_addresses,
	shift_offsets,
)
from dynamic_call_tree_resolution.vsa.memory import (
	Context,
	load_value,
	register_destination,
	scaled_addresses,
	store_at,
	store_value,
	weakened,
)
from dynamic_call_tree_resolution.vsa.state import (
	State,
	Store,
	copy_register,
	escaping,
	frame_based,
	set_offsets,
	set_register,
	stack_read,
	top_written,
	unknown_memory,
)

if TYPE_CHECKING:
	from capstone import ArmCsOperand, CsInsn


def _arm_operands(instruction: CsInsn) -> tuple[ArmCsOperand, ...]:
	return tuple(cast("ArmCsOperand", operand) for operand in instruction.operands)


def apply_arm(context: Context, instruction: CsInsn, state: State) -> State:
	executed = _apply_unconditional(context, instruction, arm_mnemonic(instruction), state)
	return weakened(context, state, executed) if arm_predicated(instruction) else executed


def _apply_unconditional(
	context: Context, instruction: CsInsn, base_mnemonic: str, state: State
) -> State:
	if base_mnemonic in ARM_MULTIPLE_STORES:
		return _arm_store_multiple(context, instruction, base_mnemonic, state)
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
	if base_mnemonic in ARM_LOADS:
		load_destination, load_source = _arm_operands(instruction)[0], instruction.operands[1]
		value = load_value(context, state, instruction, load_source)
		if len(instruction.operands) == 3:
			offset = instruction.operands[2]
			updated = _advance(state, load_source.mem.base, offset.imm)
			return set_register(updated, load_destination.reg, value)
		if instruction.writeback and load_source.mem.base != load_destination.reg:
			return set_register(
				_advance(state, load_source.mem.base, load_source.mem.disp),
				load_destination.reg,
				value,
			)
		return set_register(state, load_destination.reg, value)
	if base_mnemonic in ARM_STORE_WIDTHS:
		return _arm_store(context, instruction, base_mnemonic, state)
	if base_mnemonic.startswith(("st", "vst")):
		return top_written(
			instruction,
			escaping(
				unknown_memory(state),
				tuple(
					operand.reg
					for operand in _arm_operands(instruction)
					if operand.type == arm_const.ARM_OP_REG
				),
			),
		)
	if base_mnemonic in ARM_MOVES:
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


def _arm_store(context: Context, instruction: CsInsn, base_mnemonic: str, state: State) -> State:
	operands = _arm_operands(instruction)
	memory_index = next(
		index for index, operand in enumerate(operands) if operand.type == arm_const.ARM_OP_MEM
	)
	stored = store_value(
		context,
		escaping(
			state,
			tuple(
				operand.reg
				for operand in operands[
					1 if base_mnemonic in ARM_EXCLUSIVE_STORES else 0 : memory_index
				]
			),
		),
		instruction,
		instruction.operands[memory_index],
		Store(
			words=tuple(
				lookup(state.registers, operand.reg)
				for operand in operands[: ARM_STORED_REGISTERS.get(base_mnemonic, 0)]
			),
			width=ARM_STORE_WIDTHS[base_mnemonic],
		),
	)
	advanced = (
		_advance(stored, operands[memory_index].mem.base, operands[-1].imm)
		if memory_index < len(operands) - 1
		else _advance(stored, operands[memory_index].mem.base, operands[memory_index].mem.disp)
		if instruction.writeback
		else stored
	)
	return (
		set_register(advanced, operands[0].reg, Top())
		if base_mnemonic in ARM_EXCLUSIVE_STORES
		else advanced
	)


def _arm_store_multiple(
	context: Context, instruction: CsInsn, base_mnemonic: str, state: State
) -> State:
	operands = _arm_operands(instruction)
	implicit_sp = base_mnemonic in ("push", "vpush")
	base_register = arm_const.ARM_REG_SP if implicit_sp else operands[0].reg
	listed = operands if implicit_sp else operands[1:]
	width = sum(
		ARM_REGISTER_BYTES.get(instruction.reg_name(operand.reg)[0], 4) for operand in listed
	)
	descending = base_mnemonic in ARM_DESCENDING_STORES
	stored = store_at(
		context,
		escaping(state, tuple(operand.reg for operand in listed)),
		register_destination(context, state, base_register, -width if descending else 0),
		Store(
			words=()
			if base_mnemonic.startswith("v")
			else tuple(lookup(state.registers, operand.reg) for operand in listed),
			width=width,
		),
	)
	if not (implicit_sp or instruction.writeback):
		return stored
	return _advance(stored, base_register, -width if descending else width)


def _advance(state: State, register: int, delta: int) -> State:
	if register == arm_const.ARM_REG_SP or register in state.sp_offsets:
		return set_offsets(
			state, register, shift_offsets(lookup(state.sp_offsets, register), delta)
		)
	return set_register(state, register, shift_addresses(lookup(state.registers, register), delta))


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
		if any(
			frame_based(state, context.program.machine, operand.reg)
			for operand in (register_operand, value_operand)
		):
			return set_offsets(state, destination.reg, Top())
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
		escaped=state.escaped,
	)
