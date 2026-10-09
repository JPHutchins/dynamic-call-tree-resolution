# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Fixpoint interpretation of one function."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from capstone import x86_const
from salix import Struct

from dynamic_call_tree_resolution.model import (
	Address,
	CallSite,
	Dead,
	InstructionFamily,
	Machine,
	Unreached,
)
from dynamic_call_tree_resolution.vsa.abi import (
	ARM_ARGUMENT_REGISTERS,
	ARM_CALLER_SAVED,
	ARM_CALLS,
	ARM_STACK_ARGUMENTS,
	EM_386_STACK_ARGUMENTS,
	REGISTER_ARGUMENTS,
	SP_REGISTERS,
	X86_64_ARGUMENT_REGISTERS,
	X86_CALLER_SAVED_32,
	X86_CALLER_SAVED_64,
	X86_CALLS,
	is_data,
	normalized,
)
from dynamic_call_tree_resolution.vsa.arm import apply_arm
from dynamic_call_tree_resolution.vsa.cfg import (
	Block,
	CallsNoReturn,
	DirectCall,
	MemorySite,
	RegisterSite,
	TailJump,
	direct_transfer,
	indirect_operand,
	is_returning_trap,
	reachable_blocks,
)
from dynamic_call_tree_resolution.vsa.conditions import Conditional, branch_taken
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
	site: Address
	"""The call instruction."""
	arguments: Mapping[int, ValueSet]
	"""By ABI argument position."""


def _clobber_caller_saved(state: State, machine: Machine) -> State:
	match machine:
		case Machine.EM_386:
			return top_registers(state, X86_CALLER_SAVED_32)
		case Machine.EM_X86_64:
			return top_registers(state, X86_CALLER_SAVED_64)
		case Machine.EM_ARM:
			return top_registers(state, ARM_CALLER_SAVED)
		case _ as unreachable:
			assert_never(unreachable)


def _called(state: State, machine: Machine) -> State:
	passed = escaping(state, REGISTER_ARGUMENTS[machine])
	return _clobber_caller_saved(
		State(
			registers=passed.registers,
			sp_offsets=passed.sp_offsets,
			stack={} if passed.escaped else passed.stack,
			globals=Writes(values={}, wild=passed.globals.wild),
			escaped=passed.escaped,
			loaded_from=passed.loaded_from,
		),
		machine,
	)


def transfer(context: Context, instruction: CsInsn, state: State) -> State:
	if is_data(instruction):
		return state
	if _calls(instruction, context.program.machine) or is_returning_trap(
		instruction, context.program.machine
	):
		return _called(state, context.program.machine)
	match context.program.machine:
		case Machine.EM_386:
			return _x86_transfer(context, instruction, state, x86_const.X86_REG_ECX)
		case Machine.EM_X86_64:
			return _x86_transfer(context, instruction, state, x86_const.X86_REG_RCX)
		case Machine.EM_ARM:
			return apply_arm(context, instruction, state)
		case _ as unreachable:
			assert_never(unreachable)


def _calls(instruction: CsInsn, machine: Machine) -> bool:
	match machine.family:
		case InstructionFamily.X86:
			return instruction.mnemonic in X86_CALLS
		case InstructionFamily.ARM:
			return instruction.id in ARM_CALLS
		case _ as unreachable:
			assert_never(unreachable)


def _x86_transfer(context: Context, instruction: CsInsn, state: State, loop_counter: int) -> State:
	if instruction.mnemonic in ("loop", "loope", "loopne"):
		return top_registers(state, (loop_counter,))
	return apply_x86(context, instruction, state)


class FunctionResult(Struct):
	"""One function's interpretation."""

	sites: tuple[CallSite, ...]
	writes: Writes
	observations: tuple[CallObservation, ...]
	"""Of direct calls."""
	indirect_observations: tuple[CallObservation, ...]
	"""Of calls through a site the analysis tracked to known functions."""


def block_states(
	context: Context, blocks: tuple[Block, ...], seed: State
) -> tuple[Mapping[Address, State], Writes]:
	"""The state on entry to each block the seed reaches, and every write on the way."""
	entry = blocks[0].start
	in_states: dict[Address, State] = {entry: seed}
	worklist = [entry]
	by_start = {block.start: block for block in blocks}
	writes = NO_WRITES
	while worklist:
		block = by_start[worklist.pop()]
		incoming = in_states[block.start]
		before_last = incoming
		for instruction in block.instructions[:-1]:
			before_last = transfer(context, instruction, before_last)
		outgoing = transfer(context, block.instructions[-1], before_last)
		writes = accumulate_writes(writes, [outgoing.globals])
		for successor in _live_successors(block, before_last):
			existing = in_states.get(successor)
			joined = join_states(existing, outgoing)
			if joined != existing:
				in_states[successor] = joined
				worklist.append(successor)
	return in_states, writes


def _live_successors(block: Block, state: State) -> tuple[Address, ...]:
	match block.condition:
		case None:
			return block.successors
		case Conditional(test=test, taken=taken, fallthrough=fallthrough):
			match branch_taken(test, state.registers):
				case None:
					return block.successors
				case True:
					return taken
				case False:
					return fallthrough
				case _ as unreachable:
					assert_never(unreachable)
		case CallsNoReturn():
			return ()
		case _ as unreachable:
			assert_never(unreachable)


def analyze_function(
	context: Context, function: Function, blocks: tuple[Block, ...], seed: State
) -> FunctionResult:
	in_states, writes = block_states(context, blocks, seed)
	reachable = reachable_blocks(blocks) if len(in_states) < len(blocks) else frozenset[Address]()
	sites = tuple(
		(block, site)
		for block in blocks
		if (
			site := _site_resolution(
				context, block, function.address, in_states.get(block.start), reachable
			)
		)
		is not None
	)
	return FunctionResult(
		sites=tuple(site for _, site in sites),
		writes=writes,
		observations=tuple(
			observation
			for block in blocks
			if (state := in_states.get(block.start)) is not None
			and (
				observation := _call_observation(
					context, normalized(function.address, context.program.machine), block, state
				)
			)
			is not None
		),
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
			return CallObservation(
				callee=callee,
				site=Address(block.instructions[-1].address),
				arguments=_call_arguments(context, state, 0),
			)
		case TailJump(target=callee):
			return CallObservation(
				callee=callee,
				site=Address(block.instructions[-1].address),
				arguments=_call_arguments(context, state, 1),
			)
		case _ as unreachable:
			assert_never(unreachable)


def _indirect_observations(
	context: Context, block: Block, site: CallSite, state: State
) -> tuple[CallObservation, ...]:
	match site.target:
		case Known(values=values):
			return tuple(
				CallObservation(
					callee=callee,
					site=Address(block.instructions[-1].address),
					arguments=_call_arguments(
						context, state, 1 if block.instructions[-1].mnemonic == "jmp" else 0
					),
				)
				for callee in sorted(
					frozenset(
						normalized(candidate, context.program.machine) for candidate in values
					)
					& context.function_starts
				)
			)
		case Top() | Unreached() | Dead():
			return ()
		case _ as unreachable:
			assert_never(unreachable)


def _call_arguments(
	context: Context, state: State, pushed_return_addresses: int
) -> Mapping[int, ValueSet]:
	match context.program.machine:
		case Machine.EM_X86_64:
			return {
				position: lookup(state.registers, register)
				for position, register in enumerate(X86_64_ARGUMENT_REGISTERS)
			}
		case Machine.EM_386:
			return {
				position: stack_read(
					state.stack,
					shift_offsets(
						lookup(state.sp_offsets, SP_REGISTERS[Machine.EM_386][0]),
						_stack_argument_offset(
							position, pushed_return_addresses, context.program.pointer_size
						),
					),
				)
				for position in range(EM_386_STACK_ARGUMENTS)
			}
		case Machine.EM_ARM:
			return {
				**{
					position: lookup(state.registers, register)
					for position, register in enumerate(ARM_ARGUMENT_REGISTERS)
				},
				**{
					len(ARM_ARGUMENT_REGISTERS) + slot: stack_read(
						state.stack,
						shift_offsets(
							lookup(state.sp_offsets, SP_REGISTERS[Machine.EM_ARM][0]),
							context.program.pointer_size * slot,
						),
					)
					for slot in range(ARM_STACK_ARGUMENTS)
				},
			}
		case _ as unreachable:
			assert_never(unreachable)


def _stack_argument_offset(position: int, pushed_return_addresses: int, pointer_size: int) -> int:
	"""How far above the stack pointer an EM_386 argument sits at a call or tail jump.

	At a ``call`` the return address is not pushed yet, so the first argument is at the
	stack pointer itself; at a tail ``jmp`` the caller's return address is still at the
	stack pointer, so every argument sits one pointer higher.

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
	context: Context,
	block: Block,
	caller_address: Address,
	state: State | None,
	reachable: frozenset[Address],
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
			target=Dead() if block.start in reachable else Unreached(),
		)
	match site_operand:
		case RegisterSite(operand=operand):
			value = lookup(state.registers, operand.reg)
			return _call_site(
				context,
				caller_address,
				instruction,
				_single(value),
				value,
				state.loaded_from.get(operand.reg),
			)
		case MemorySite(operand=operand):
			slot = _single(_site_operand_addresses(context, state, instruction, operand))
			return _call_site(
				context,
				caller_address,
				instruction,
				slot,
				load_value(context, state, instruction, operand),
				slot,
			)
		case _ as unreachable:
			assert_never(unreachable)


def _call_site(
	context: Context,
	caller_address: Address,
	instruction: CsInsn,
	slot: Address | None,
	target: ValueSet,
	loaded_from: Address | None,
) -> CallSite:
	return CallSite(
		caller_address=caller_address,
		site_address=Address(instruction.address),
		slot=slot,
		target=target,
		loaded_from=loaded_from,
		external=context.external_symbols.get(loaded_from) if loaded_from is not None else None,
	)


def _single(value: ValueSet) -> Address | None:
	match value:
		case Top():
			return None
		case Known(values=values):
			return next(iter(values)) if len(values) == 1 else None
		case _ as unreachable:
			assert_never(unreachable)
