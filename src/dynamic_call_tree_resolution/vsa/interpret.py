# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Fixpoint interpretation of one function."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from capstone import x86_const
from salix import Struct

from dynamic_call_tree_resolution.model import Address, CallSite, Machine
from dynamic_call_tree_resolution.vsa.abi import (
	ARM_ARGUMENT_REGISTERS,
	ARM_CALLER_SAVED,
	ARM_CALLS,
	EM_386_STACK_ARGUMENTS,
	REGISTER_ARGUMENTS,
	SP_REGISTERS,
	X86_64_ARGUMENT_REGISTERS,
	X86_CALLER_SAVED_32,
	X86_CALLER_SAVED_64,
	X86_CALLS,
	arm_mnemonic,
	normalized,
)
from dynamic_call_tree_resolution.vsa.arm import apply_arm
from dynamic_call_tree_resolution.vsa.cfg import (
	Block,
	DirectCall,
	MemorySite,
	RegisterSite,
	TailJump,
	direct_transfer,
	indirect_operand,
	is_returning_trap,
)
from dynamic_call_tree_resolution.vsa.lattice import Known, Top, ValueSet, lookup, shift_offsets
from dynamic_call_tree_resolution.vsa.memory import (
	Context,
	accumulate_writes,
	load_value,
	memory_addresses,
)
from dynamic_call_tree_resolution.vsa.state import (
	NO_WRITES,
	State,
	Writes,
	escaping,
	frame_based,
	join_states,
	stack_read,
	top_registers,
)
from dynamic_call_tree_resolution.vsa.x86 import apply_x86

if TYPE_CHECKING:
	from collections.abc import Mapping

	from capstone import CsInsn, CsOperand

	from dynamic_call_tree_resolution.model import Function


class CallObservation(Struct):
	"""One call to a known function."""

	callee: Address
	arguments: Mapping[int, ValueSet]
	"""By ABI argument position."""


def _clobber_caller_saved(state: State, machine: Machine) -> State:
	registers = (
		X86_CALLER_SAVED_32
		if machine is Machine.EM_386
		else X86_CALLER_SAVED_64
		if machine is Machine.EM_X86_64
		else ARM_CALLER_SAVED
	)
	return top_registers(state, registers)


def _called(state: State, machine: Machine) -> State:
	passed = escaping(state, REGISTER_ARGUMENTS[machine])
	return _clobber_caller_saved(
		State(
			registers=passed.registers,
			sp_offsets=passed.sp_offsets,
			stack={} if passed.escaped else passed.stack,
			globals=Writes(values={}, wild=passed.globals.wild),
			escaped=passed.escaped,
		),
		machine,
	)


def _transfer(context: Context, instruction: CsInsn, state: State) -> State:
	machine = context.program.machine
	if instruction.mnemonic == ".byte":
		return state
	if (
		instruction.mnemonic in X86_CALLS
		if machine.is_x86
		else arm_mnemonic(instruction) in ARM_CALLS
	) or is_returning_trap(instruction, machine):
		return _called(state, machine)
	if instruction.mnemonic in ("loop", "loope", "loopne"):
		return top_registers(
			state, (x86_const.X86_REG_ECX if machine is Machine.EM_386 else x86_const.X86_REG_RCX,)
		)
	if machine.is_x86:
		return apply_x86(context, instruction, state)
	return apply_arm(context, instruction, state)


class FunctionResult(Struct):
	"""One function's interpretation."""

	sites: tuple[CallSite, ...]
	writes: Writes
	observations: tuple[CallObservation, ...]
	"""Of direct calls."""
	indirect_observations: tuple[CallObservation, ...]
	"""Of calls through a site the analysis tracked to known functions."""


def analyze_function(
	context: Context, function: Function, blocks: tuple[Block, ...], seed: State
) -> FunctionResult:
	entry = blocks[0].start
	in_states: dict[Address, State] = {entry: seed}
	worklist = [entry]
	by_start = {block.start: block for block in blocks}
	writes = NO_WRITES
	while worklist:
		block = by_start[worklist.pop()]
		incoming = in_states[block.start]
		outgoing = incoming
		for instruction in block.instructions:
			outgoing = _transfer(context, instruction, outgoing)
		writes = accumulate_writes(writes, [outgoing.globals])
		for successor in block.successors:
			existing = in_states.get(successor)
			joined = join_states(existing, outgoing)
			if joined != existing:
				in_states[successor] = joined
				worklist.append(successor)
	observations: list[CallObservation] = []
	for block in blocks:
		state = in_states.get(block.start)
		if state is None:
			continue
		observation = _call_observation(
			context, normalized(function.address, context.program.machine), block, state
		)
		if observation is not None:
			observations.append(observation)
	sites = tuple(
		(block, site)
		for block in blocks
		if (site := _site_resolution(context, block, function.address, in_states.get(block.start)))
		is not None
	)
	return FunctionResult(
		sites=tuple(site for _, site in sites),
		writes=writes,
		observations=tuple(observations),
		indirect_observations=tuple(
			observation
			for block, site in sites
			if (state := in_states.get(block.start)) is not None
			for observation in _indirect_observations(context, block, site, state)
		),
	)


def _call_observation(
	context: Context, function_start: Address, block: Block, state: State
) -> CallObservation | None:
	match direct_transfer(
		block.instructions[-1], context.program.machine, function_start, context.function_starts
	):
		case None:
			return None
		case DirectCall(target=callee):
			return CallObservation(callee=callee, arguments=_call_arguments(context, state, 0))
		case TailJump(target=callee):
			return CallObservation(callee=callee, arguments=_call_arguments(context, state, 1))
		case _ as unreachable:
			assert_never(unreachable)


def _indirect_observations(
	context: Context, block: Block, site: CallSite, state: State
) -> tuple[CallObservation, ...]:
	return tuple(
		CallObservation(
			callee=callee,
			arguments=_call_arguments(
				context, state, 1 if block.instructions[-1].mnemonic == "jmp" else 0
			),
		)
		for callee in sorted(
			frozenset(
				normalized(candidate, context.program.machine) for candidate in site.candidates
			)
			& context.function_starts
		)
	)


def _call_arguments(
	context: Context, state: State, pushed_return_addresses: int
) -> Mapping[int, ValueSet]:
	machine = context.program.machine
	if machine is Machine.EM_X86_64:
		return {
			position: lookup(state.registers, register)
			for position, register in enumerate(X86_64_ARGUMENT_REGISTERS)
		}
	if machine is Machine.EM_386:
		offsets = lookup(state.sp_offsets, SP_REGISTERS[machine][0])
		return {
			position: stack_read(
				state.stack,
				shift_offsets(
					offsets,
					_stack_argument_offset(
						position, pushed_return_addresses, context.program.pointer_size
					),
				),
			)
			for position in range(EM_386_STACK_ARGUMENTS)
		}
	return {
		position: lookup(state.registers, register)
		for position, register in enumerate(ARM_ARGUMENT_REGISTERS)
	}


def _stack_argument_offset(position: int, pushed_return_addresses: int, pointer_size: int) -> int:
	"""How far above the stack pointer an EM_386 argument sits at a call or tail jump.

	At a ``call`` the return address is not pushed yet, so the first argument is at the
	stack pointer itself; at a tail ``jmp``, the caller's return address is below it.

	>>> _stack_argument_offset(0, 0, 4), _stack_argument_offset(0, 1, 4)
	(0, 4)
	>>> _stack_argument_offset(2, 0, 4)
	8
	"""
	return pointer_size * (position + pushed_return_addresses)


def _site_operand_addresses(
	context: Context, state: State, instruction: CsInsn, operand: CsOperand
) -> ValueSet:
	memory = operand.mem
	if memory.base == x86_const.X86_REG_RIP:
		return Known(
			values=frozenset({Address(instruction.address + instruction.size + memory.disp)})
		)
	if frame_based(state, context.program.machine, memory.base):
		return Top()
	return memory_addresses(context, state, memory)


def _site_resolution(
	context: Context, block: Block, caller_address: Address, state: State | None
) -> CallSite | None:
	instruction = block.instructions[-1]
	site_operand = indirect_operand(instruction, context.program.machine)
	if site_operand is None:
		return None
	if state is None:
		return CallSite(
			caller_address=caller_address,
			site_address=Address(instruction.address),
			slot=None,
			candidates=frozenset(),
		)
	match site_operand:
		case RegisterSite(operand=operand):
			value = lookup(state.registers, operand.reg)
			return CallSite(
				caller_address=caller_address,
				site_address=Address(instruction.address),
				slot=_single(value),
				candidates=_candidates(value),
			)
		case MemorySite(operand=operand):
			return CallSite(
				caller_address=caller_address,
				site_address=Address(instruction.address),
				slot=_single(_site_operand_addresses(context, state, instruction, operand)),
				candidates=_candidates(load_value(context, state, instruction, operand)),
			)
		case _ as unreachable:
			assert_never(unreachable)


def _single(value: ValueSet) -> Address | None:
	match value:
		case Top():
			return None
		case Known(values=values):
			return next(iter(values)) if len(values) == 1 else None
		case _ as unreachable:
			assert_never(unreachable)


def _candidates(value: ValueSet) -> frozenset[Address]:
	match value:
		case Top():
			return frozenset()
		case Known(values=values):
			return values
		case _ as unreachable:
			assert_never(unreachable)
