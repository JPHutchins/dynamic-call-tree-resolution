# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Whole-function value-set analysis of indirect call sites."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Hashable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Final, Literal, cast

from capstone import (
	CS_ARCH_ARM,
	CS_ARCH_X86,
	CS_GRP_BRANCH_RELATIVE,
	CS_GRP_CALL,
	CS_GRP_JUMP,
	CS_GRP_RET,
	CS_MODE_32,
	CS_MODE_64,
	CS_MODE_THUMB,
	Cs,
	arm_const,
	x86_const,
)
from salix import Struct

from dynamic_call_tree_resolution.model import Address, CallSite, Machine, aligned, thumb_twin
from dynamic_call_tree_resolution.points_to import memory_at, read_pointer

if TYPE_CHECKING:
	from collections.abc import Iterator, Mapping

	from capstone import ArmCsOperand, CsInsn, CsMemOperand, CsOperand

	from dynamic_call_tree_resolution.model import Function, Program

_K_BOUND: Final = 64

_GLOBAL_ROUNDS: Final = 3

_MAX_ROUNDS: Final = 8

_DISASSEMBLERS: Final[Mapping[Machine, tuple[int, int]]] = {
	Machine.EM_X86_64: (CS_ARCH_X86, CS_MODE_64),
	Machine.EM_386: (CS_ARCH_X86, CS_MODE_32),
	Machine.EM_ARM: (CS_ARCH_ARM, CS_MODE_THUMB),
}

_SP_REGISTERS: Final[Mapping[Machine, tuple[int, ...]]] = {
	Machine.EM_X86_64: (x86_const.X86_REG_RSP, x86_const.X86_REG_RBP),
	Machine.EM_386: (x86_const.X86_REG_ESP, x86_const.X86_REG_EBP),
	Machine.EM_ARM: (arm_const.ARM_REG_SP,),
}

_X86_TRANSFERS: Final = (
	"call",
	"jmp",
	"ret",
	"retf",
	"iret",
	"loop",
	"loope",
	"loopne",
	"int",
	"int3",
	"syscall",
	"ud2",
	"hlt",
)
_ARM_TRANSFERS: Final = ("bl", "blx", "bx", "b", "pop", "svc", "bkpt", "udf", "tbb", "tbh")
_X86_CALLS: Final = ("call",)
_ARM_CALLS: Final = ("bl", "blx")
_ARM_CONDITIONAL: Final = frozenset(
	{
		"beq",
		"bne",
		"bcs",
		"bcc",
		"bmi",
		"bpl",
		"bvs",
		"bvc",
		"bhi",
		"bls",
		"bge",
		"blt",
		"bgt",
		"ble",
		"bhs",
		"blo",
		"cbz",
		"cbnz",
	}
)
_X86_CALLER_SAVED_64: Final = (
	x86_const.X86_REG_RAX,
	x86_const.X86_REG_RCX,
	x86_const.X86_REG_RDX,
	x86_const.X86_REG_RSI,
	x86_const.X86_REG_RDI,
	x86_const.X86_REG_R8,
	x86_const.X86_REG_R9,
	x86_const.X86_REG_R10,
	x86_const.X86_REG_R11,
)
_X86_CALLER_SAVED_32: Final = (
	x86_const.X86_REG_EAX,
	x86_const.X86_REG_ECX,
	x86_const.X86_REG_EDX,
	x86_const.X86_REG_ESI,
	x86_const.X86_REG_EDI,
)
_ARM_CALLER_SAVED: Final = (
	arm_const.ARM_REG_R0,
	arm_const.ARM_REG_R1,
	arm_const.ARM_REG_R2,
	arm_const.ARM_REG_R3,
	arm_const.ARM_REG_R12,
	arm_const.ARM_REG_LR,
)
_X86_MOVES: Final = ("mov", "movabs")
_ARM_LOADS: Final = ("ldr", "ldr.w", "ldr.n")
_ARM_MOVES: Final = ("mov", "movs", "mov.w")
_X86_64_ARGUMENT_REGISTERS: Final = (
	x86_const.X86_REG_RDI,
	x86_const.X86_REG_RSI,
	x86_const.X86_REG_RDX,
	x86_const.X86_REG_RCX,
	x86_const.X86_REG_R8,
	x86_const.X86_REG_R9,
)
_ARM_ARGUMENT_REGISTERS: Final = (
	arm_const.ARM_REG_R0,
	arm_const.ARM_REG_R1,
	arm_const.ARM_REG_R2,
	arm_const.ARM_REG_R3,
)
_EM_386_STACK_ARGUMENTS: Final = 8

type ValueSet = frozenset[Address] | None
type OffsetSet = frozenset[int] | None


class State(Struct):
	"""Abstract state at one block entry; an absent entry is Top, an empty set Bottom."""

	registers: Mapping[int, frozenset[Address]]
	sp_offsets: Mapping[int, frozenset[int]]
	"""Stack-pointer registers, as offsets from the entry stack pointer."""
	stack: Mapping[int, frozenset[Address]]
	"""Frame slots, by entry-frame offset."""
	globals: Mapping[int, frozenset[Address]]
	"""Values written to program-global addresses."""


class _Block(Struct):
	start: Address
	instructions: tuple[CsInsn, ...]
	successors: tuple[Address, ...]


class _CallObservation(Struct):
	callee: Address
	arguments: Mapping[int, ValueSet]
	"""By ABI argument position."""


class _Context(Struct):
	program: Program
	object_spans: tuple[tuple[int, int], ...]
	object_starts: tuple[int, ...]
	global_writes: Mapping[int, frozenset[Address]]
	function_starts: frozenset[Address]


def analyze(program: Program) -> tuple[CallSite, ...]:
	disassembler = Cs(*_DISASSEMBLERS[program.machine])
	disassembler.detail = True
	disassembler.skipdata = True
	blocks_by_function = _blocks_by_function(program, disassembler)
	functions = tuple(blocks_by_function.values())
	_prewarm_instructions(functions, program.machine)
	seeds: dict[Address, State] = {}
	global_writes: dict[int, frozenset[Address]] = {}
	stable_write_rounds = 0
	with ThreadPoolExecutor() as executor:
		for _ in range(_MAX_ROUNDS):
			context = _context_for(program, global_writes)
			observations: list[_CallObservation] = []
			written: list[Mapping[int, frozenset[Address]]] = []
			for result in executor.map(
				partial(_round_analysis, program, context, seeds), functions
			):
				written.append(result.writes)
				observations.extend(result.observations)
			next_seeds = dict(seeds)
			for observation in observations:
				next_seeds[observation.callee] = _join_seeds(
					next_seeds.get(observation.callee),
					_seed_from_observation(observation, program),
				)
			next_writes = _accumulate_writes(global_writes, written)
			if next_seeds == seeds and next_writes == global_writes:
				break
			seeds_changed = next_seeds != seeds
			seeds, global_writes = next_seeds, next_writes
			if seeds_changed:
				stable_write_rounds = 0
			else:
				stable_write_rounds += 1
				if stable_write_rounds >= _GLOBAL_ROUNDS:
					break
		final_context = _context_for(program, global_writes)
		return tuple(
			site
			for sites in executor.map(
				partial(_final_sites, program, final_context, seeds), functions
			)
			for site in sites
		)


def _round_analysis(
	program: Program,
	context: _Context,
	seeds: Mapping[Address, State],
	function_blocks: tuple[Function, tuple[_Block, ...]],
) -> _FunctionResult:
	function, blocks = function_blocks
	seed = _seed(seeds.get(_normalized(function.address, program.machine)), program.machine)
	return _analyze_function(context, function, blocks, seed)


def _final_sites(
	program: Program,
	context: _Context,
	seeds: Mapping[Address, State],
	function_blocks: tuple[Function, tuple[_Block, ...]],
) -> tuple[CallSite, ...]:
	function, blocks = function_blocks
	return _analyze_function(
		context,
		function,
		blocks,
		_seed(seeds.get(_normalized(function.address, program.machine)), program.machine),
	).sites


def _prewarm_instructions(
	functions: tuple[tuple[Function, tuple[_Block, ...]], ...], machine: Machine
) -> None:
	for _, blocks in functions:
		for block in blocks:
			for instruction in block.instructions:
				if instruction.mnemonic == ".byte":
					continue
				for operand in instruction.operands:
					if machine is Machine.EM_ARM:
						_ = cast("ArmCsOperand", operand).shift
					else:
						_ = operand.mem


def _function_address(function: Function) -> Address:
	return function.address


def _blocks_by_function(
	program: Program, disassembler: Cs
) -> dict[Address, tuple[Function, tuple[_Block, ...]]]:
	seen: set[Address] = set()
	blocks: dict[Address, tuple[Function, tuple[_Block, ...]]] = {}
	for function in sorted(program.functions.values(), key=_function_address):
		# ARM symbol addresses carry the Thumb bit; strip it so the code
		# decodes from the aligned start, and skip the symbol/DWARF twin.
		start = aligned(function.address) if program.machine is Machine.EM_ARM else function.address
		if start in seen:
			continue
		code = memory_at(program, start, function.size)
		if not code:
			continue
		seen.add(start)
		instructions = tuple(disassembler.disasm(code, start))
		blocks[start] = (function, _build_blocks(instructions, program.machine))
	return blocks


def address_taken(program: Program) -> frozenset[Address]:
	functions = frozenset(program.functions)
	return frozenset(
		address
		for address in functions
		if any(
			address.to_bytes(program.pointer_size, program.byte_order) in data
			for data in program.sections.values()
		)
	) | frozenset(Address(value) for value in _computed_addresses(program) if value in functions)


def _computed_addresses(program: Program) -> Iterator[int]:
	disassembler = Cs(*_DISASSEMBLERS[program.machine])
	disassembler.detail = True
	disassembler.skipdata = True
	for function in program.functions.values():
		start = _normalized(function.address, program.machine)
		instructions = tuple(
			instruction
			for instruction in disassembler.disasm(memory_at(program, start, function.size), start)
			if instruction.id != 0 and not any(instruction.group(group) for group in _BRANCH_GROUPS)
		)
		for index, instruction in enumerate(instructions):
			yield from (
				_arm_addresses(instruction, instructions[:index])
				if program.machine is Machine.EM_ARM
				else _x86_addresses(instruction)
			)


_BRANCH_GROUPS: Final = (CS_GRP_CALL, CS_GRP_JUMP, CS_GRP_RET, CS_GRP_BRANCH_RELATIVE)


def _arm_addresses(instruction: CsInsn, preceding: tuple[CsInsn, ...]) -> Iterator[int]:
	operands = instruction.operands
	yield from (operand.imm for operand in operands if operand.type == arm_const.ARM_OP_IMM)
	match instruction.mnemonic.split(".")[0], operands:
		case "adr", [_, offset]:
			yield thumb_twin(Address(_aligned_pc(instruction) + offset.imm))
		case (("add" | "addw"), [_, base, offset]) if base.reg == arm_const.ARM_REG_PC:
			yield thumb_twin(Address(_aligned_pc(instruction) + offset.imm))
		case (("sub" | "subw"), [_, base, offset]) if base.reg == arm_const.ARM_REG_PC:
			yield thumb_twin(Address(_aligned_pc(instruction) - offset.imm))
		case "movt", [destination, high]:
			yield from _movt_addresses(high.imm, destination.reg, preceding)
		case _:
			pass


def _aligned_pc(instruction: CsInsn) -> int:
	return (instruction.address + 4) & ~3


def _movt_addresses(high: int, register: int, preceding: tuple[CsInsn, ...]) -> Iterator[int]:
	low = next(
		(
			earlier.operands[1].imm
			for earlier in reversed(preceding)
			if earlier.mnemonic.split(".")[0] == "movw" and earlier.operands[0].reg == register
		),
		None,
	)
	if low is not None:
		yield (high << 16) | (low & 0xFFFF)


def _x86_addresses(instruction: CsInsn) -> Iterator[int]:
	for operand in instruction.operands:
		if operand.type == x86_const.X86_OP_IMM:
			yield operand.imm
		if (
			instruction.mnemonic == "lea"
			and operand.type == x86_const.X86_OP_MEM
			and operand.mem.base == x86_const.X86_REG_RIP
		):
			yield instruction.address + instruction.size + operand.mem.disp


def _normalized(address: Address, machine: Machine) -> Address:
	return aligned(address) if machine is Machine.EM_ARM else address


def _seed(base: State | None, machine: Machine) -> State:
	if base is None:
		return State(
			registers={},
			sp_offsets={_SP_REGISTERS[machine][0]: frozenset({0})},
			stack={},
			globals={},
		)
	return State(registers=base.registers, sp_offsets=base.sp_offsets, stack=base.stack, globals={})


def _seed_from_observation(observation: _CallObservation, program: Program) -> State:
	registers: dict[int, frozenset[Address]] = {}
	stack: dict[int, frozenset[Address]] = {}
	for position, value in observation.arguments.items():
		if value is None:
			continue
		if program.machine is Machine.EM_386:
			stack[program.pointer_size * (position + 1)] = value
		elif program.machine is Machine.EM_X86_64:
			registers[_X86_64_ARGUMENT_REGISTERS[position]] = value
		else:
			registers[_ARM_ARGUMENT_REGISTERS[position]] = value
	return State(
		registers=registers,
		sp_offsets={_SP_REGISTERS[program.machine][0]: frozenset({0})},
		stack=stack,
		globals={},
	)


def _accumulate_writes(
	current: Mapping[int, frozenset[Address]], written: list[Mapping[int, frozenset[Address]]]
) -> dict[int, frozenset[Address]]:
	accumulated = dict(current)
	for function_writes in written:
		for address, values in function_writes.items():
			existing = accumulated.get(address)
			if existing is None:
				accumulated[address] = values
				continue
			combined = existing | values
			if len(combined) > _K_BOUND:
				accumulated.pop(address, None)
			else:
				accumulated[address] = combined
	return accumulated


def _context_for(program: Program, global_writes: Mapping[int, frozenset[Address]]) -> _Context:
	spans = tuple(
		sorted(
			(object_.address, object_.address + object_.size)
			for object_ in program.objects.values()
			if object_.size > 0
		)
	)
	return _Context(
		program=program,
		object_spans=spans,
		object_starts=tuple(span[0] for span in spans),
		global_writes=global_writes,
		function_starts=frozenset(
			_normalized(function.address, program.machine)
			for function in program.functions.values()
		),
	)


def _span_at(context: _Context, address: Address) -> tuple[int, int] | None:
	index = bisect_right(context.object_starts, address) - 1
	if index < 0:
		return None
	span = context.object_spans[index]
	return span if span[0] <= address < span[1] else None


def _pointer_value(context: _Context, address: Address) -> Address | None:
	span = _span_at(context, address)
	return read_pointer(context.program, address, span[1] if span is not None else None)


def _object_slots(context: _Context, start: Address, stride: int) -> ValueSet:
	span = _span_at(context, start)
	if span is None or stride <= 0:
		return None
	_, object_end = span
	count = (object_end - start) // stride
	if count <= 0:
		return None
	return frozenset(Address(start + index * stride) for index in range(count))


def _image_value(context: _Context, state: State, addresses: ValueSet) -> ValueSet:
	if addresses is None:
		return None
	values: set[Address] = set()
	for address in addresses:
		written = state.globals.get(address)
		if written is None:
			written = context.global_writes.get(address)
		if written is not None:
			values.update(written)
		value = _pointer_value(context, address)
		if value is not None:
			values.add(value)
		elif written is None and _span_at(context, address) is not None:
			return None
	return None if len(values) > _K_BOUND else frozenset(values)


def _join_sets[T: Hashable](current: frozenset[T], incoming: frozenset[T]) -> frozenset[T] | None:
	combined = current | incoming
	return None if len(combined) > _K_BOUND else combined


def _join_maps[T: Hashable](
	current: Mapping[int, frozenset[T]], incoming: Mapping[int, frozenset[T]]
) -> dict[int, frozenset[T]]:
	joined: dict[int, frozenset[T]] = {}
	for key in set(current) | set(incoming):
		if key not in current or key not in incoming:
			continue
		value = _join_sets(current[key], incoming[key])
		if value is not None:
			joined[key] = value
	return joined


def _join_seeds(current: State | None, incoming: State) -> State:
	if current is None:
		return incoming
	registers = dict(current.registers)
	for register, value in incoming.registers.items():
		if register not in current.registers:
			registers[register] = value
			continue
		joined = _join_sets(current.registers[register], value)
		if joined is not None:
			registers[register] = joined
	stack = dict(current.stack)
	for offset, value in incoming.stack.items():
		if offset not in current.stack:
			stack[offset] = value
			continue
		joined = _join_sets(current.stack[offset], value)
		if joined is not None:
			stack[offset] = joined
	return State(
		registers=registers, sp_offsets=current.sp_offsets, stack=stack, globals=current.globals
	)


def _join_states(current: State | None, incoming: State) -> State:
	if current is None:
		return incoming
	return State(
		registers=_join_maps(current.registers, incoming.registers),
		sp_offsets=_join_maps(current.sp_offsets, incoming.sp_offsets),
		stack=_join_maps(current.stack, incoming.stack),
		globals=_join_maps(current.globals, incoming.globals),
	)


def _put_value[T: Hashable](
	mapping: Mapping[int, frozenset[T]], key: int, value: frozenset[T] | None
) -> dict[int, frozenset[T]]:
	copied = dict(mapping)
	if value is None:
		copied.pop(key, None)
	else:
		copied[key] = value
	return copied


def _map_set[T: Hashable, U: Hashable](
	values: frozenset[T] | None, function: Callable[[T], U]
) -> frozenset[U] | None:
	return None if values is None else frozenset(function(value) for value in values)


def _shift_addresses(values: ValueSet, delta: int) -> ValueSet:
	return _map_set(values, lambda address: Address(address + delta))


def _shift_offsets(values: OffsetSet, delta: int) -> OffsetSet:
	return _map_set(values, lambda offset: offset + delta)


def _indexed(base: ValueSet, index: ValueSet, scale: int, disp: int) -> ValueSet:
	if base is None or index is None:
		return None
	addresses = frozenset(Address(b + i * scale + disp) for b in base for i in index)
	return None if len(addresses) > _K_BOUND else addresses


def _indexed_offsets(
	base: frozenset[int], index: frozenset[Address], scale: int, disp: int
) -> OffsetSet:
	offsets = frozenset(b + i * scale + disp for b in base for i in index)
	return None if len(offsets) > _K_BOUND else offsets


def _set_register(state: State, register: int, value: ValueSet) -> State:
	sp_offsets = dict(state.sp_offsets)
	sp_offsets.pop(register, None)
	return State(
		registers=_put_value(state.registers, register, value),
		sp_offsets=sp_offsets,
		stack=state.stack,
		globals=state.globals,
	)


def _set_offsets(state: State, register: int, value: OffsetSet) -> State:
	registers = dict(state.registers)
	registers.pop(register, None)
	return State(
		registers=registers,
		sp_offsets=_put_value(state.sp_offsets, register, value),
		stack=state.stack,
		globals=state.globals,
	)


def _top_registers(state: State, registers: tuple[int, ...]) -> State:
	remaining = {
		register: value for register, value in state.registers.items() if register not in registers
	}
	return State(
		registers=remaining, sp_offsets=state.sp_offsets, stack=state.stack, globals=state.globals
	)


def _top_written(instruction: CsInsn, state: State) -> State:
	written = set(instruction.regs_access()[1])
	return State(
		registers={
			register: value
			for register, value in state.registers.items()
			if register not in written
		},
		sp_offsets={
			register: value
			for register, value in state.sp_offsets.items()
			if register not in written
		},
		stack=state.stack,
		globals=state.globals,
	)


def _stack_read(stack: Mapping[int, frozenset[Address]], offsets: OffsetSet) -> ValueSet:
	if offsets is None:
		return None
	values: set[Address] = set()
	for offset in offsets:
		value = stack.get(offset)
		if value is None:
			return None
		values.update(value)
	return None if len(values) > _K_BOUND else frozenset(values)


def _union_write(
	mapping: Mapping[int, frozenset[Address]], keys: OffsetSet, value: ValueSet
) -> dict[int, frozenset[Address]]:
	written = dict(mapping)
	if keys is None:
		return written
	for offset in keys:
		if value is None:
			written.pop(offset, None)
			continue
		existing = written.get(offset)
		joined = _join_sets(existing, value) if existing is not None else value
		if joined is None:
			written.pop(offset, None)
		else:
			written[offset] = joined
	return written


def _stack_offsets(state: State, memory: CsMemOperand, base_register: int) -> OffsetSet:
	offsets = state.sp_offsets.get(base_register)
	if memory.index == 0:
		return _shift_offsets(offsets, memory.disp)
	index = state.registers.get(memory.index)
	if offsets is None or index is None:
		return None
	return _indexed_offsets(offsets, index, memory.scale, memory.disp)


def _scaled_addresses(
	context: _Context, base: ValueSet, index: ValueSet, scale: int, disp: int
) -> ValueSet:
	if index is None:
		if base is not None and len(base) == 1:
			(base_address,) = base
			slots = _object_slots(context, Address(base_address + disp), scale)
			if slots is not None:
				return slots
		return None
	return _indexed(base, index, scale, disp)


def _memory_addresses(context: _Context, state: State, memory: CsMemOperand) -> ValueSet:
	base = frozenset({Address(0)}) if memory.base == 0 else state.registers.get(memory.base)
	if memory.index == 0:
		return _shift_addresses(base, memory.disp)
	return _scaled_addresses(
		context, base, state.registers.get(memory.index), memory.scale, memory.disp
	)


def _load_value(
	context: _Context, state: State, instruction: CsInsn, operand: CsOperand
) -> ValueSet:
	machine = context.program.machine
	memory = operand.mem
	if machine.is_x86:
		if memory.base == x86_const.X86_REG_RIP:
			address = Address(instruction.address + instruction.size + memory.disp)
			return _image_value(context, state, frozenset({address}))
		if memory.base in _SP_REGISTERS[machine] or memory.base in state.sp_offsets:
			return _stack_read(state.stack, _stack_offsets(state, memory, memory.base))
		return _image_value(context, state, _memory_addresses(context, state, memory))
	if memory.base == arm_const.ARM_REG_PC:
		address = Address(((instruction.address + 4) & ~3) + memory.disp)
		return _image_value(context, state, frozenset({address}))
	if memory.base == arm_const.ARM_REG_SP or memory.base in state.sp_offsets:
		return _stack_read(state.stack, _stack_offsets(state, memory, memory.base))
	return _image_value(context, state, _memory_addresses(context, state, memory))


def _store_value(
	context: _Context, state: State, instruction: CsInsn, operand: CsOperand, value: ValueSet
) -> State:
	machine = context.program.machine
	memory = operand.mem
	if machine.is_x86:
		if memory.base in _SP_REGISTERS[machine] or memory.base in state.sp_offsets:
			return State(
				registers=state.registers,
				sp_offsets=state.sp_offsets,
				stack=_union_write(state.stack, _stack_offsets(state, memory, memory.base), value),
				globals=state.globals,
			)
		addresses = _store_addresses(context, state, instruction, memory)
		return State(
			registers=state.registers,
			sp_offsets=state.sp_offsets,
			stack=state.stack,
			globals=_union_write(state.globals, addresses, value),
		)
	if memory.base == arm_const.ARM_REG_SP or memory.base in state.sp_offsets:
		return State(
			registers=state.registers,
			sp_offsets=state.sp_offsets,
			stack=_union_write(state.stack, _stack_offsets(state, memory, memory.base), value),
			globals=state.globals,
		)
	addresses = _store_addresses(context, state, instruction, memory)
	return State(
		registers=state.registers,
		sp_offsets=state.sp_offsets,
		stack=state.stack,
		globals=_union_write(state.globals, addresses, value),
	)


def _store_addresses(
	context: _Context, state: State, instruction: CsInsn, memory: CsMemOperand
) -> ValueSet:
	machine = context.program.machine
	if machine.is_x86 and memory.base == x86_const.X86_REG_RIP:
		return frozenset({Address(instruction.address + instruction.size + memory.disp)})
	return _memory_addresses(context, state, memory)


def _copy_register(state: State, destination: int, source: int) -> State:
	if destination == source:
		return state
	if source in state.sp_offsets:
		return _set_offsets(state, destination, state.sp_offsets[source])
	return _set_register(state, destination, state.registers.get(source))


class _BranchTarget(Struct):
	target: int
	conditional: bool


def _branch_target(instruction: CsInsn, machine: Machine) -> _BranchTarget | None:
	if machine.is_x86:
		if instruction.mnemonic == "jmp":
			conditional = False
		elif instruction.mnemonic.startswith("j") or instruction.mnemonic in (
			"loop",
			"loope",
			"loopne",
		):
			conditional = True
		else:
			return None
		operand = instruction.operands[0]
		if operand.type != x86_const.X86_OP_IMM:
			return None
		return _BranchTarget(target=operand.imm, conditional=conditional)
	base = instruction.mnemonic.split(".")[0]
	if base == "b":
		conditional = False
	elif base in _ARM_CONDITIONAL:
		conditional = True
	else:
		return None
	operand = instruction.operands[1 if base in ("cbz", "cbnz") else 0]
	if operand.type != arm_const.ARM_OP_IMM:
		return None  # pragma: no cover
	return _BranchTarget(target=operand.imm, conditional=conditional)


def _call_target(instruction: CsInsn, machine: Machine) -> int | None:
	if machine.is_x86:
		if instruction.mnemonic != "call":
			return None
		operand = instruction.operands[0]
		if operand.type != x86_const.X86_OP_IMM:
			return None
		return operand.imm
	if instruction.mnemonic.split(".")[0] not in ("bl", "blx"):
		return None
	operand = instruction.operands[0]
	if operand.type != arm_const.ARM_OP_IMM:
		return None
	return operand.imm


def _is_control_transfer(instruction: CsInsn, machine: Machine) -> bool:
	if machine.is_x86:
		return instruction.mnemonic in _X86_TRANSFERS
	return instruction.mnemonic.split(".")[0] in _ARM_TRANSFERS


def _site_operand(
	instruction: CsInsn, machine: Machine
) -> tuple[Literal["memory", "register"], CsOperand] | None:
	if instruction.mnemonic == ".byte":
		return None
	if not instruction.operands:
		return None
	operand = instruction.operands[0]
	mnemonic = instruction.mnemonic
	if machine.is_x86:
		if mnemonic in ("call", "jmp"):
			if operand.type == x86_const.X86_OP_MEM:
				return "memory", operand
			if operand.type == x86_const.X86_OP_REG:
				return "register", operand
		return None
	if mnemonic == "blx" and operand.type == arm_const.ARM_OP_REG:
		return "register", operand
	if mnemonic == "bx" and operand.reg not in (arm_const.ARM_REG_LR, arm_const.ARM_REG_PC):
		return "register", operand
	return None


def _successors(
	instruction: CsInsn, machine: Machine, by_address: Mapping[int, int], next_address: int | None
) -> tuple[Address, ...]:
	branch = _branch_target(instruction, machine)
	if branch is not None:
		successors: list[Address] = []
		if branch.conditional and next_address is not None:
			successors.append(Address(next_address))
		if branch.target in by_address:
			successors.append(Address(branch.target))
		return tuple(successors)
	fallthrough = (Address(next_address),) if next_address is not None else ()
	if instruction.mnemonic == ".byte":
		return fallthrough
	if _is_control_transfer(instruction, machine):
		if instruction.mnemonic in (_X86_CALLS if machine.is_x86 else _ARM_CALLS):
			return fallthrough
		if (
			instruction.mnemonic.split(".")[0] == "pop"
			and arm_const.ARM_REG_PC not in instruction.regs_access()[1]
		):
			return fallthrough
		return ()
	raise AssertionError  # every block-ending instruction matches an arm above


def _block_ends(instruction: CsInsn, machine: Machine) -> bool:
	return (
		instruction.mnemonic == ".byte"
		or _branch_target(instruction, machine) is not None
		or _is_control_transfer(instruction, machine)
	)


def _build_blocks(instructions: tuple[CsInsn, ...], machine: Machine) -> tuple[_Block, ...]:
	by_address = {instruction.address: index for index, instruction in enumerate(instructions)}
	targets = frozenset(
		branch.target
		for instruction in instructions
		if (branch := _branch_target(instruction, machine)) is not None
	)
	blocks: list[_Block] = []
	current: list[CsInsn] = []
	for index, instruction in enumerate(instructions):
		if (
			instruction.address in targets
			or _site_operand(instruction, machine) is not None
			or _call_target(instruction, machine) is not None
		) and current:
			blocks.append(
				_Block(
					start=Address(current[0].address),
					instructions=tuple(current),
					successors=(Address(instruction.address),),
				)
			)
			current = []
		current.append(instruction)
		if _block_ends(instruction, machine):
			next_address = (
				instructions[index + 1].address if index + 1 < len(instructions) else None
			)
			blocks.append(
				_Block(
					start=Address(current[0].address),
					instructions=tuple(current),
					successors=_successors(instruction, machine, by_address, next_address),
				)
			)
			current = []
	if current:
		blocks.append(
			_Block(start=Address(current[0].address), instructions=tuple(current), successors=())
		)
	return tuple(blocks)


def _clobber_caller_saved(state: State, machine: Machine) -> State:
	registers = (
		_X86_CALLER_SAVED_32
		if machine is Machine.EM_386
		else _X86_CALLER_SAVED_64
		if machine is Machine.EM_X86_64
		else _ARM_CALLER_SAVED
	)
	return _top_registers(state, registers)


def _transfer(context: _Context, instruction: CsInsn, state: State) -> State:
	machine = context.program.machine
	if instruction.mnemonic == ".byte":
		return state
	if instruction.mnemonic in (_X86_CALLS if machine.is_x86 else _ARM_CALLS):
		return _clobber_caller_saved(state, machine)
	if instruction.mnemonic in ("loop", "loope", "loopne"):
		return _top_registers(
			state, (x86_const.X86_REG_ECX if machine is Machine.EM_386 else x86_const.X86_REG_RCX,)
		)
	if machine.is_x86:
		return _apply_x86(context, instruction, state)
	return _apply_arm(context, instruction, state)


def _apply_x86(context: _Context, instruction: CsInsn, state: State) -> State:
	machine = context.program.machine
	if instruction.mnemonic in _X86_MOVES:
		destination, source = instruction.operands
		match (destination.type, source.type):
			case (x86_const.X86_OP_REG, x86_const.X86_OP_REG):
				return _copy_register(state, destination.reg, source.reg)
			case (x86_const.X86_OP_REG, x86_const.X86_OP_IMM):
				return _set_register(state, destination.reg, frozenset({Address(source.imm)}))
			case (x86_const.X86_OP_REG, x86_const.X86_OP_MEM):
				return _set_register(
					state, destination.reg, _load_value(context, state, instruction, source)
				)
			case (x86_const.X86_OP_MEM, x86_const.X86_OP_REG):
				return _store_value(
					context, state, instruction, destination, state.registers.get(source.reg)
				)
			case (x86_const.X86_OP_MEM, x86_const.X86_OP_IMM):
				return _store_value(
					context, state, instruction, destination, frozenset({Address(source.imm)})
				)
			case _:
				return _top_written(instruction, state)  # pragma: no cover
	if instruction.mnemonic == "lea":
		destination, source = instruction.operands
		memory = source.mem
		if memory.base == x86_const.X86_REG_RIP:
			address = Address(instruction.address + instruction.size + memory.disp)
			return _set_register(state, destination.reg, frozenset({address}))
		if memory.base in _SP_REGISTERS[machine]:
			return _set_offsets(state, destination.reg, _stack_offsets(state, memory, memory.base))
		return _set_register(state, destination.reg, _memory_addresses(context, state, memory))
	if instruction.mnemonic in ("inc", "dec"):
		destination = instruction.operands[0]
		if destination.type != x86_const.X86_OP_REG:  # pragma: no branch
			return _top_written(instruction, state)
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
			return _top_written(instruction, state)
		delta = source.imm if instruction.mnemonic == "add" else -source.imm
		return _shift_x86_destination(state, machine, destination.reg, destination.reg, delta)
	if instruction.mnemonic == "xor":
		destination, source = instruction.operands
		if (
			destination.type == x86_const.X86_OP_REG
			and source.type == x86_const.X86_OP_REG
			and destination.reg == source.reg
		):
			return _set_register(state, destination.reg, frozenset({Address(0)}))
		return _top_written(instruction, state)
	if instruction.mnemonic == "push":
		operand = instruction.operands[0]
		value = (
			frozenset({Address(operand.imm)})
			if operand.type == x86_const.X86_OP_IMM
			else state.registers.get(operand.reg)
			if operand.type == x86_const.X86_OP_REG
			else _load_value(context, state, instruction, operand)
		)
		return _push(state, _SP_REGISTERS[machine][0], context.program.pointer_size, value)
	if instruction.mnemonic == "pop":
		operand = instruction.operands[0]
		return _pop(state, _SP_REGISTERS[machine][0], context.program.pointer_size, operand.reg)
	return _top_written(instruction, state)


def _shift_x86_destination(
	state: State, machine: Machine, destination: int, source: int, delta: int
) -> State:
	if destination in _SP_REGISTERS[machine]:
		return _set_offsets(
			state, destination, _shift_offsets(state.sp_offsets.get(destination), delta)
		)
	return _set_register(state, destination, _shift_addresses(state.registers.get(source), delta))


def _push(state: State, sp: int, pointer_size: int, value: ValueSet) -> State:
	offsets = _shift_offsets(state.sp_offsets.get(sp), -pointer_size)
	return State(
		registers=state.registers,
		sp_offsets=_put_value(state.sp_offsets, sp, offsets),
		stack=_union_write(state.stack, offsets, value),
		globals=state.globals,
	)


def _pop(state: State, sp: int, pointer_size: int, destination: int) -> State:
	offsets = state.sp_offsets.get(sp)
	return State(
		registers=_put_value(state.registers, destination, _stack_read(state.stack, offsets)),
		sp_offsets=_put_value(state.sp_offsets, sp, _shift_offsets(offsets, pointer_size)),
		stack=state.stack,
		globals=state.globals,
	)


def _arm_operands(instruction: CsInsn) -> tuple[ArmCsOperand, ...]:
	return tuple(cast("ArmCsOperand", operand) for operand in instruction.operands)


def _apply_arm(context: _Context, instruction: CsInsn, state: State) -> State:
	base_mnemonic = instruction.mnemonic.split(".")[0]
	if base_mnemonic == "push":
		return _arm_push(instruction, state)
	if base_mnemonic == "pop":
		return _arm_pop(instruction, state)
	if base_mnemonic in ("movw", "movt"):
		destination, source = _arm_operands(instruction)
		if source.type != arm_const.ARM_OP_IMM:  # pragma: no branch
			return _top_written(instruction, state)  # pragma: no cover
		if base_mnemonic == "movw":
			return _set_register(state, destination.reg, frozenset({Address(source.imm)}))
		return _set_register(
			state,
			destination.reg,
			_map_set(
				state.registers.get(destination.reg),
				lambda low: Address((source.imm << 16) | (low & 0xFFFF)),
			),
		)
	if instruction.mnemonic in _ARM_LOADS:
		load_destination, load_source = _arm_operands(instruction)[0], instruction.operands[1]
		value = _load_value(context, state, instruction, load_source)
		if len(instruction.operands) == 3:
			offset = instruction.operands[2]
			updated = _advance_post_index(context, state, load_source, offset.imm)
			return _set_register(updated, load_destination.reg, value)
		return _set_register(state, load_destination.reg, value)
	if instruction.mnemonic in ("str", "str.w"):
		store_source, store_destination = _arm_operands(instruction)[0], instruction.operands[1]
		if store_source.type != arm_const.ARM_OP_REG:  # pragma: no branch
			return _top_written(instruction, state)  # pragma: no cover
		stored = _store_value(
			context, state, instruction, store_destination, state.registers.get(store_source.reg)
		)
		if len(instruction.operands) == 3:
			return _advance_post_index(
				context, stored, store_destination, instruction.operands[2].imm
			)
		return stored
	if instruction.mnemonic in _ARM_MOVES:
		destination, source = _arm_operands(instruction)
		match source.type:
			case arm_const.ARM_OP_REG:
				return _copy_register(state, destination.reg, source.reg)
			case arm_const.ARM_OP_IMM:
				return _set_register(state, destination.reg, frozenset({Address(source.imm)}))
			case _:
				return _top_written(instruction, state)  # pragma: no cover
	if base_mnemonic in ("add", "adds", "sub", "subs"):
		return _arm_arithmetic(context, instruction, state, base_mnemonic)
	return _top_written(instruction, state)


def _advance_post_index(context: _Context, state: State, operand: CsOperand, delta: int) -> State:
	base = operand.mem.base
	if base == arm_const.ARM_REG_SP:
		return _set_offsets(state, base, _shift_offsets(state.sp_offsets.get(base), delta))
	return _set_register(state, base, _shift_addresses(state.registers.get(base), delta))


def _arm_shift_scale(operand: ArmCsOperand) -> int | None:
	if operand.shift.type == arm_const.ARM_SFT_LSL:
		return 1 << operand.shift.value
	if operand.shift.type == arm_const.ARM_SFT_INVALID:
		return 1
	return None


def _arm_arithmetic(
	context: _Context, instruction: CsInsn, state: State, base_mnemonic: str
) -> State:
	operands = _arm_operands(instruction)
	destination = operands[0]
	sign = 1 if base_mnemonic.startswith("add") else -1
	if len(operands) == 2:
		source = operands[1]
		if source.type != arm_const.ARM_OP_IMM:  # pragma: no branch
			return _top_written(instruction, state)  # pragma: no cover
		return _arm_shift_destination(state, destination.reg, destination.reg, sign * source.imm)
	register_operand, value_operand = operands[1], operands[2]
	if value_operand.type == arm_const.ARM_OP_IMM:
		return _arm_shift_destination(
			state, destination.reg, register_operand.reg, sign * value_operand.imm
		)
	if value_operand.type == arm_const.ARM_OP_REG:
		scale = _arm_shift_scale(value_operand)
		if scale is None:
			return _top_written(instruction, state)
		return _set_register(
			state,
			destination.reg,
			_scaled_addresses(
				context,
				state.registers.get(register_operand.reg),
				state.registers.get(value_operand.reg),
				scale,
				0,
			),
		)
	return _top_written(instruction, state)  # pragma: no cover


def _arm_shift_destination(state: State, destination: int, source: int, delta: int) -> State:
	if destination == arm_const.ARM_REG_SP:
		return _set_offsets(
			state, destination, _shift_offsets(state.sp_offsets.get(destination), delta)
		)
	if source == arm_const.ARM_REG_SP:
		return _set_offsets(state, destination, _shift_offsets(state.sp_offsets.get(source), delta))
	return _set_register(state, destination, _shift_addresses(state.registers.get(source), delta))


def _arm_push(instruction: CsInsn, state: State) -> State:
	operands = _arm_operands(instruction)
	registers = tuple(operand.reg for operand in operands if operand.reg != arm_const.ARM_REG_SP)
	offsets = _shift_offsets(state.sp_offsets.get(arm_const.ARM_REG_SP), -4 * len(registers))
	stack = state.stack
	for index, register in enumerate(registers):
		stack = _union_write(
			stack, _shift_offsets(offsets, 4 * index), state.registers.get(register)
		)
	return State(
		registers=state.registers,
		sp_offsets=_put_value(state.sp_offsets, arm_const.ARM_REG_SP, offsets),
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
	offsets = state.sp_offsets.get(arm_const.ARM_REG_SP)
	registers_map = state.registers
	for index, register in enumerate(registers):
		registers_map = _put_value(
			registers_map, register, _stack_read(state.stack, _shift_offsets(offsets, 4 * index))
		)
	return State(
		registers=registers_map,
		sp_offsets=_put_value(
			state.sp_offsets, arm_const.ARM_REG_SP, _shift_offsets(offsets, 4 * len(operands))
		),
		stack=state.stack,
		globals=state.globals,
	)


class _FunctionResult(Struct):
	sites: tuple[CallSite, ...]
	writes: Mapping[int, frozenset[Address]]
	observations: tuple[_CallObservation, ...]


def _analyze_function(
	context: _Context, function: Function, blocks: tuple[_Block, ...], seed: State
) -> _FunctionResult:
	entry = blocks[0].start
	in_states: dict[Address, State] = {entry: seed}
	worklist = [entry]
	by_start = {block.start: block for block in blocks}
	writes: dict[int, frozenset[Address]] = {}
	while worklist:
		block = by_start[worklist.pop()]
		incoming = in_states[block.start]
		outgoing = incoming
		for instruction in block.instructions:
			outgoing = _transfer(context, instruction, outgoing)
		writes = _accumulate_writes(writes, [outgoing.globals])
		for successor in block.successors:
			existing = in_states.get(successor)
			joined = _join_states(existing, outgoing)
			if joined != existing:
				in_states[successor] = joined
				worklist.append(successor)
	observations: list[_CallObservation] = []
	for block in blocks:
		state = in_states.get(block.start)
		if state is None:
			continue
		observation = _call_observation(context, block, state)
		if observation is not None:
			observations.append(observation)
	return _FunctionResult(
		sites=tuple(
			site
			for block in blocks
			if (
				site := _site_resolution(
					context, block, function.address, in_states.get(block.start)
				)
			)
			is not None
		),
		writes=writes,
		observations=tuple(observations),
	)


def _call_observation(context: _Context, block: _Block, state: State) -> _CallObservation | None:
	instruction = block.instructions[-1]
	target = _call_target(instruction, context.program.machine)
	if target is None:
		return None
	callee = _normalized(Address(target), context.program.machine)
	if callee not in context.function_starts:
		return None
	return _CallObservation(callee=callee, arguments=_call_arguments(context, state, instruction))


def _call_arguments(context: _Context, state: State, instruction: CsInsn) -> Mapping[int, ValueSet]:
	machine = context.program.machine
	if machine is Machine.EM_X86_64:
		return {
			position: state.registers.get(register)
			for position, register in enumerate(_X86_64_ARGUMENT_REGISTERS)
		}
	if machine is Machine.EM_386:
		# at the call instruction the return address is not pushed yet, so
		# the first argument sits at the top of the caller's stack
		offsets = state.sp_offsets.get(_SP_REGISTERS[machine][0])
		return {
			position: _stack_read(
				state.stack, _shift_offsets(offsets, context.program.pointer_size * position)
			)
			for position in range(_EM_386_STACK_ARGUMENTS)
		}
	return {
		position: state.registers.get(register)
		for position, register in enumerate(_ARM_ARGUMENT_REGISTERS)
	}


def _site_operand_addresses(
	context: _Context, state: State, instruction: CsInsn, operand: CsOperand
) -> ValueSet:
	memory = operand.mem
	if memory.base == x86_const.X86_REG_RIP:
		return frozenset({Address(instruction.address + instruction.size + memory.disp)})
	if memory.base in _SP_REGISTERS[context.program.machine] or memory.base in state.sp_offsets:
		return None
	return _memory_addresses(context, state, memory)


def _site_resolution(
	context: _Context, block: _Block, caller_address: Address, state: State | None
) -> CallSite | None:
	instruction = block.instructions[-1]
	site_operand = _site_operand(instruction, context.program.machine)
	if site_operand is None:
		return None
	if state is None:
		return CallSite(
			caller_address=caller_address,
			site_address=Address(instruction.address),
			slot=None,
			candidates=frozenset(),
		)
	kind, operand = site_operand
	if kind == "register":
		value = state.registers.get(operand.reg)
		return CallSite(
			caller_address=caller_address,
			site_address=Address(instruction.address),
			slot=Address(next(iter(value))) if value is not None and len(value) == 1 else None,
			candidates=frozenset() if value is None else value,
		)
	addresses = _site_operand_addresses(context, state, instruction, operand)
	value = _load_value(context, state, instruction, operand)
	return CallSite(
		caller_address=caller_address,
		site_address=Address(instruction.address),
		slot=Address(next(iter(addresses)))
		if addresses is not None and len(addresses) == 1
		else None,
		candidates=frozenset() if value is None else value,
	)
