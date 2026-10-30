# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Transfer functions for x86 instructions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from capstone import x86_const
from salix import Struct

from dynamic_call_tree_resolution.model import Address, Machine
from dynamic_call_tree_resolution.vsa.abi import (
	SP_REGISTERS,
	X86_MEMORY_READERS,
	X86_MOVES,
	X86_REGISTERS,
	X86_REPEATS,
	X86_UNBOUNDED_STORES,
)
from dynamic_call_tree_resolution.vsa.lattice import (
	Known,
	OffsetSet,
	Top,
	ValueSet,
	lookup,
	map_set,
	put_value,
	shift_addresses,
	shift_offsets,
)
from dynamic_call_tree_resolution.vsa.memory import (
	Context,
	Frame,
	anywhere,
	load_value,
	loaded,
	memory_addresses,
	memory_destination,
	stack_offsets,
	store_at,
	store_value,
)
from dynamic_call_tree_resolution.vsa.state import (
	State,
	Store,
	escaping,
	frame_based,
	set_loaded,
	set_offsets,
	set_register,
	stack_read,
	top_written,
	without_facts,
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
				return _copy(state, machine, destination.reg, source.reg)
			case (x86_const.X86_OP_REG, x86_const.X86_OP_IMM):
				return _write(
					state, machine, destination.reg, Known(values=frozenset({Address(source.imm)}))
				)
			case (x86_const.X86_OP_REG, x86_const.X86_OP_MEM):
				word = loaded(context, state, instruction, source, context.program.pointer_size)
				return _write(state, machine, destination.reg, word.value, word.origin)
			case (x86_const.X86_OP_MEM, x86_const.X86_OP_REG):
				return store_value(
					context,
					escaping(state, (source.reg,), lowest_tracked=context.upward_frame_writes),
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
			case _:  # pragma: no cover
				return top_written(
					instruction,
					state,
					_ROOT_IDS[machine],
					lowest_tracked=context.upward_frame_writes,
				)
	if instruction.mnemonic == "lea":
		destination, source = _x86_operands(instruction)
		memory = source.mem
		if memory.base == x86_const.X86_REG_RIP:
			address = Address(instruction.address + instruction.size + memory.disp)
			return _write(state, machine, destination.reg, Known(values=frozenset({address})))
		if frame_based(state, machine, memory.base):
			return _write_offsets(
				state, machine, destination.reg, stack_offsets(context, state, source)
			)
		return _write(state, machine, destination.reg, memory_addresses(context, state, source))
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
			return _write(state, machine, destination.reg, Known(values=frozenset({Address(0)})))
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
		return _push(
			context,
			escaping(
				state,
				(operand.reg,) if operand.type == x86_const.X86_OP_REG else (),
				lowest_tracked=context.upward_frame_writes,
			),
			SP_REGISTERS[machine][0],
			value,
		)
	if instruction.mnemonic == "pop":
		operand = _x86_operands(instruction)[0]
		if operand.type != x86_const.X86_OP_REG:
			return _clobbered(context, instruction, state)
		return _pop(state, machine, context.program.pointer_size, operand.reg)
	return _clobbered(context, instruction, state)


def _clobbered(context: Context, instruction: CsInsn, state: State) -> State:
	prefixes_and_name = instruction.mnemonic.split()
	operands = _x86_operands(instruction)
	if (
		not operands
		or operands[0].type != x86_const.X86_OP_MEM
		or prefixes_and_name[-1] in X86_MEMORY_READERS
	):
		return top_written(
			instruction,
			state,
			_ROOT_IDS[context.program.machine],
			lowest_tracked=context.upward_frame_writes,
		)
	destination = memory_destination(context, state, instruction, operands[0])
	return top_written(
		instruction,
		store_at(
			context,
			escaping(
				state,
				tuple(operand.reg for operand in operands if operand.type == x86_const.X86_OP_REG),
				lowest_tracked=context.upward_frame_writes,
			),
			anywhere(destination)
			if prefixes_and_name[0] in X86_REPEATS or prefixes_and_name[-1] in X86_UNBOUNDED_STORES
			else destination,
			Store(words=(), width=operands[0].size),
			Address(instruction.address),
		),
		_ROOT_IDS[context.program.machine],
		lowest_tracked=context.upward_frame_writes,
	)


def _shift_x86_destination(
	state: State, machine: Machine, destination: int, source: int, delta: int
) -> State:
	if frame_based(state, machine, destination):
		return _write_offsets(
			state, machine, destination, shift_offsets(lookup(state.sp_offsets, destination), delta)
		)
	return _write(
		state, machine, destination, shift_addresses(lookup(state.registers, source), delta)
	)


def _push(context: Context, state: State, sp: int, value: ValueSet) -> State:
	pointer_size = context.program.pointer_size
	offsets = shift_offsets(lookup(state.sp_offsets, sp), -pointer_size)
	return set_offsets(
		store_at(
			context, state, Frame(offsets=offsets), Store(words=(value,), width=pointer_size), None
		),
		sp,
		offsets,
	)


class _RegisterRoot(Struct):
	"""The register a write lands in."""

	root: int
	bits: int
	"""The width of the part written, or 0 for a register outside every family."""


def _pop(state: State, machine: Machine, pointer_size: int, destination: int) -> State:
	sp = SP_REGISTERS[machine][0]
	offsets = lookup(state.sp_offsets, sp)
	written = _root(machine, destination)
	return State(
		registers=put_value(
			state.registers,
			written.root,
			stack_read(state.stack, offsets) if written.bits in (0, _ROOT_BITS[machine]) else Top(),
		),
		sp_offsets=put_value(state.sp_offsets, sp, shift_offsets(offsets, pointer_size)),
		stack=state.stack,
		globals=state.globals,
		escaped=state.escaped,
		loaded_from=without_facts(state.loaded_from, (written.root,)),
		strided=without_facts(state.strided, (written.root,)),
	)


_ROOT_BITS: Final = {Machine.EM_X86_64: 64, Machine.EM_386: 32}


_ROOTS: Final = {
	machine: {
		register: _RegisterRoot(
			root=family.wide if machine is Machine.EM_X86_64 else family.narrow, bits=family.bits
		)
		for register, family in X86_REGISTERS.items()
	}
	for machine in (Machine.EM_X86_64, Machine.EM_386)
}
_ROOT_IDS: Final = {
	machine: {register: found.root for register, found in roots.items()}
	for machine, roots in _ROOTS.items()
}


def _root(machine: Machine, register: int) -> _RegisterRoot:
	return (
		found
		if (found := _ROOTS[machine].get(register)) is not None
		else _RegisterRoot(root=register, bits=0)
	)


def _write(
	state: State,
	machine: Machine,
	register: int,
	value: ValueSet,
	source: Address | None = None,
) -> State:
	written = _root(machine, register)
	return (
		set_loaded(state, written.root, value, source)
		if written.bits in (0, _ROOT_BITS[machine])
		else set_loaded(state, written.root, map_set(value, _low_32), source)
		if written.bits == 32
		else set_register(state, written.root, Top())
	)


def _low_32(value: Address) -> Address:
	return Address(value & 0xFFFFFFFF)


def _write_offsets(state: State, machine: Machine, register: int, offsets: OffsetSet) -> State:
	written = _root(machine, register)
	return set_offsets(
		state, written.root, offsets if written.bits in (0, _ROOT_BITS[machine]) else Top()
	)


def _copy(state: State, machine: Machine, destination: int, source: int) -> State:
	if source in state.sp_offsets:
		return _write_offsets(state, machine, destination, Known(values=state.sp_offsets[source]))
	return _write(
		state,
		machine,
		destination,
		lookup(state.registers, source),
		state.loaded_from.get(source),
	)
