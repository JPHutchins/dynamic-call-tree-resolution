# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Control-flow graphs of function code."""

from __future__ import annotations

from functools import partial, reduce
from itertools import groupby, takewhile
from typing import TYPE_CHECKING, Final, assert_never, cast

from capstone import arm_const, x86_const
from salix import Struct

from dynamic_call_tree_resolution.model import Address, InstructionFamily, Machine, aligned
from dynamic_call_tree_resolution.points_to import (
	in_writable_memory,
	instruction_runs,
	instruction_set_at,
	memory_at,
)
from dynamic_call_tree_resolution.vsa.abi import (
	ARM_CALLS,
	ARM_CONDITIONAL,
	ARM_LOAD_WIDTHS,
	ARM_RETURNING_TRAPS,
	ARM_TRANSFERS,
	X86_CALLS,
	X86_RETURNING_TRAPS,
	X86_TRANSFERS,
	arm_mnemonic,
	arm_predicated,
	disassemblers,
	normalized,
	program_counter,
)

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Mapping

	from capstone import ArmCsOperand, Cs, CsInsn, CsOperand

	from dynamic_call_tree_resolution.model import Function, InstructionSet, Program


_WINDOW: Final = 16
_MAX_CASES: Final = 1024
_WORD_MASK: Final = 0xFFFF_FFFF
_NOTHING_KNOWN: Final[Mapping[int, int]] = {}


class Block(Struct):
	"""One basic block of a function's code."""

	start: Address
	instructions: tuple[CsInsn, ...]
	successors: tuple[Address, ...]


def _function_address(function: Function) -> Address:
	return function.address


def control_flow_graphs(program: Program) -> dict[Address, tuple[Function, tuple[Block, ...]]]:
	decoders = disassemblers()
	return {
		start: graph
		for start, functions in groupby(
			sorted(program.functions.values(), key=_function_address),
			key=partial(_code_start, program.machine),
		)
		if (graph := _first_graph(program, decoders, start, functions)) is not None
	}


def _code_start(machine: Machine, function: Function) -> Address:
	return normalized(function.address, machine)


def _first_graph(
	program: Program,
	decoders: Mapping[InstructionSet, Cs],
	start: Address,
	functions: Iterable[Function],
) -> tuple[Function, tuple[Block, ...]] | None:
	return next(
		filter(
			None, (function_graph(program, decoders, start, function) for function in functions)
		),
		None,
	)


def function_graph(
	program: Program, decoders: Mapping[InstructionSet, Cs], start: Address, function: Function
) -> tuple[Function, tuple[Block, ...]] | None:
	runs = instruction_runs(program, start, function.size)
	if not any(code for _, code in runs):
		return None
	decoded = tuple(
		_decoded(
			decoders[instruction_set_at(program, address)],
			program,
			address,
			code,
			(start, start + function.size),
		)
		for address, code in runs
	)
	return (
		function,
		_build_blocks(
			tuple(instruction for run, _ in decoded for instruction in run),
			program.machine,
			{address: table for _, tables in decoded for address, table in tables.items()},
		),
	)


class _JumpTable(Struct):
	"""The case addresses a bounded dispatch reaches."""

	targets: tuple[Address, ...]
	skip: int | None
	"""The end of a table that follows the dispatch, which the decoder skips."""
	guarded: tuple[int, ...]
	"""Instructions no branch may enter without skipping the bound or a value the dispatch reads."""


class _Case(Struct):
	"""Where a dispatch goes for one index."""

	target: Address
	table: tuple[int, int] | None
	"""The bytes the dispatch read the target from."""


def _decoded(
	disassembler: Cs, program: Program, start: int, code: bytes, function: tuple[int, int]
) -> tuple[tuple[CsInsn, ...], Mapping[int, _JumpTable | None]]:
	instructions: list[CsInsn] = []
	dispatches: dict[int, _JumpTable | None] = {}
	offset = 0
	while offset < len(code):
		for instruction in disassembler.disasm(code[offset:], start + offset):
			instructions.append(instruction)
			if _dispatches(instruction, program.machine):
				table = _jump_table(instructions, program, function)
				dispatches[instruction.address] = table
				if table is not None and table.skip is not None:
					offset = table.skip - start
					break
		else:
			break
	return tuple(instructions), dispatches


def _dispatches(instruction: CsInsn, machine: Machine) -> bool:
	match machine.family:
		case InstructionFamily.X86:
			return False
		case InstructionFamily.ARM:
			return _arm_dispatches(instruction)
		case _ as unreachable:
			assert_never(unreachable)


def _arm_dispatches(instruction: CsInsn) -> bool:
	match arm_mnemonic(instruction):
		case "tbb" | "tbh":
			return True
		case "ldr":
			return _writes_pc(instruction) and instruction.operands[1].mem.index != 0
		case "add":
			return _writes_pc(instruction)
		case _:
			return False


def _writes_pc(instruction: CsInsn) -> bool:
	return (
		bool(instruction.operands)
		and instruction.operands[0].type == arm_const.ARM_OP_REG
		and instruction.operands[0].reg == arm_const.ARM_REG_PC
	)


def _jump_table(
	instructions: list[CsInsn], program: Program, function: tuple[int, int]
) -> _JumpTable | None:
	guard = _window_start(instructions, len(instructions) - 1) - 1
	if (
		guard < 1
		or arm_predicated(instructions[-1])
		or not _bounds(instructions[guard - 1], instructions[guard])
		or instructions[guard - 1].operands[1].imm >= _MAX_CASES
	):
		return None
	return next(
		(
			table
			for start in range(guard - 1, _window_start(instructions, guard - 1) - 1, -1)
			if (table := _bounded(instructions, start, guard, program, function)) is not None
		),
		None,
	)


def _window_start(instructions: list[CsInsn], end: int) -> int:
	floor = max(end - _WINDOW, 0)
	return next(
		(
			index + 1
			for index in range(end - 1, floor - 1, -1)
			if _block_ends(instructions[index], Machine.EM_ARM)
		),
		floor,
	)


def _bounds(compare: CsInsn, branch: CsInsn) -> bool:
	return (
		arm_mnemonic(compare) == "cmp"
		and not arm_predicated(compare)
		and compare.operands[1].type == arm_const.ARM_OP_IMM
		and arm_mnemonic(branch) == "b"
		and branch.cc == arm_const.ARM_CC_HI
	)


def _bounded(
	instructions: list[CsInsn],
	start: int,
	guard: int,
	program: Program,
	function: tuple[int, int],
) -> _JumpTable | None:
	dispatch = instructions[-1]
	compare = instructions[guard - 1]
	known = reduce(partial(_step, program), instructions[start : guard - 1], _NOTHING_KNOWN)
	cases = tuple(
		case
		for case in takewhile(
			_is_case,
			(
				_case(
					program,
					dispatch,
					reduce(
						partial(_step, program),
						instructions[guard + 1 : -1],
						{**known, compare.operands[0].reg: index},
					),
				)
				for index in range(compare.operands[1].imm + 1)
			),
		)
		if case is not None
	)
	if len(cases) <= compare.operands[1].imm or not all(
		function[0] <= case.target < function[1] for case in cases
	):
		return None
	tables = tuple(case.table for case in cases if case.table is not None)
	end = max((high for _, high in tables), default=0)
	return _JumpTable(
		targets=tuple(case.target for case in cases),
		skip=end + end % 2
		if dispatch.address + dispatch.size
		<= min((low for low, _ in tables), default=-1)
		<= dispatch.address + dispatch.size + 2
		else None,
		guarded=tuple(instruction.address for instruction in instructions[start + 1 :]),
	)


def _is_case(case: _Case | None) -> bool:
	return case is not None


def _case(program: Program, dispatch: CsInsn, registers: Mapping[int, int]) -> _Case | None:
	match arm_mnemonic(dispatch), _arm_operands(dispatch):
		case ("tbb" | "tbh") as mnemonic, [table]:
			return _read_case(
				program,
				_address(registers, table, dispatch.address + 4),
				ARM_LOAD_WIDTHS["ldrb" if mnemonic == "tbb" else "ldrh"],
				partial(_halfwords_past, dispatch.address + 4),
			)
		case "ldr", [_, table]:
			return _read_case(
				program,
				_address(registers, table, _base(program, dispatch, registers, table)),
				ARM_LOAD_WIDTHS["ldr"],
				_absolute_target,
			)
		case "add", [_, base, offset] if (
			base.reg == arm_const.ARM_REG_PC
			and (value := _operand_value(registers, offset)) is not None
		):
			return _Case(target=Address(_pc_relative(program, dispatch) + value), table=None)
		case _:
			return None


def _read_case(
	program: Program,
	address: int | None,
	width: int,
	target: Callable[[int], Address],
) -> _Case | None:
	entry = None if address is None else _read(program, address, width)
	return (
		None
		if address is None or entry is None
		else _Case(target=target(entry), table=(address, address + width))
	)


def _halfwords_past(base: int, entry: int) -> Address:
	return Address(base + 2 * entry)


def _absolute_target(entry: int) -> Address:
	return aligned(Address(entry))


def _step(program: Program, registers: Mapping[int, int], instruction: CsInsn) -> Mapping[int, int]:
	value = (
		None
		if arm_predicated(instruction) or instruction.writeback
		else _computed(program, registers, instruction)
	)
	written = frozenset(instruction.regs_access()[1])
	kept = {register: known for register, known in registers.items() if register not in written}
	return kept if value is None else {**kept, instruction.operands[0].reg: value & _WORD_MASK}


def _computed(program: Program, registers: Mapping[int, int], instruction: CsInsn) -> int | None:
	match arm_mnemonic(instruction), _arm_operands(instruction):
		case ("ldr" | "ldrb" | "ldrh") as mnemonic, [_, source]:
			address = _address(registers, source, _base(program, instruction, registers, source))
			return None if address is None else _read(program, address, ARM_LOAD_WIDTHS[mnemonic])
		case "adr", [_, offset]:
			return _pc_relative(program, instruction) + offset.imm
		case (("add" | "adds" | "addw"), [_, base, offset]) if (
			base.reg == arm_const.ARM_REG_PC and offset.type == arm_const.ARM_OP_IMM
		):
			return _pc_relative(program, instruction) + offset.imm
		case (("add" | "adds" | "addw"), [_, augend, addend]):
			return _sum(registers, augend, addend)
		case (("mov" | "movs" | "movw" | "lsl" | "lsls"), [_, source]):
			return _operand_value(registers, source)
		case "movt", [destination, high] if (low := registers.get(destination.reg)) is not None:
			return (high.imm << 16) | (low & 0xFFFF)
		case _:
			return None


def _pc_relative(program: Program, instruction: CsInsn) -> int:
	return program_counter(instruction, instruction_set_at(program, Address(instruction.address)))


def _sum(registers: Mapping[int, int], augend: ArmCsOperand, addend: ArmCsOperand) -> int | None:
	left = _operand_value(registers, augend)
	right = _operand_value(registers, addend)
	return None if left is None or right is None else left + right


def _operand_value(registers: Mapping[int, int], operand: ArmCsOperand) -> int | None:
	return (
		operand.imm
		if operand.type == arm_const.ARM_OP_IMM
		else _shifted(registers.get(operand.reg), operand)
		if operand.type == arm_const.ARM_OP_REG
		else None
	)


def _shifted(value: int | None, operand: ArmCsOperand) -> int | None:
	if value is None:
		return None
	match operand.shift.type:
		case arm_const.ARM_SFT_INVALID:
			return value
		case arm_const.ARM_SFT_LSL:
			return value << operand.shift.value
		case _:
			return None


def _base(
	program: Program, instruction: CsInsn, registers: Mapping[int, int], operand: ArmCsOperand
) -> int | None:
	return (
		_pc_relative(program, instruction)
		if operand.mem.base == arm_const.ARM_REG_PC
		else registers.get(operand.mem.base)
	)


def _address(registers: Mapping[int, int], operand: ArmCsOperand, base: int | None) -> int | None:
	index = 0 if operand.mem.index == 0 else _shifted(registers.get(operand.mem.index), operand)
	return (
		None
		if base is None or index is None or operand.subtracted
		else (base + index + operand.mem.disp) & _WORD_MASK
	)


def _read(program: Program, address: int, width: int) -> int | None:
	data = memory_at(program, Address(address), width)
	return (
		int.from_bytes(data, program.byte_order)
		if len(data) == width and not in_writable_memory(program, Address(address))
		else None
	)


def _arm_operands(instruction: CsInsn) -> tuple[ArmCsOperand, ...]:
	return tuple(cast("ArmCsOperand", operand) for operand in instruction.operands)


class _BranchTarget(Struct):
	target: int
	conditional: bool


def branch_target(instruction: CsInsn, machine: Machine) -> _BranchTarget | None:
	match machine.family:
		case InstructionFamily.X86:
			return _x86_branch_target(instruction)
		case InstructionFamily.ARM:
			return _arm_branch_target(instruction)
		case _ as unreachable:
			assert_never(unreachable)


def _x86_branch_target(instruction: CsInsn) -> _BranchTarget | None:
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


def _arm_branch_target(instruction: CsInsn) -> _BranchTarget | None:
	base = instruction.mnemonic.split(".")[0]
	if base == "b":
		conditional = False
	elif base in ARM_CONDITIONAL:
		conditional = True
	else:
		return None
	operand = instruction.operands[1 if base in ("cbz", "cbnz") else 0]
	if operand.type != arm_const.ARM_OP_IMM:
		return None  # pragma: no cover
	return _BranchTarget(target=operand.imm, conditional=conditional)


class DirectCall(Struct):
	"""A direct call to a function."""

	target: Address


class TailJump(Struct):
	"""A direct branch into another function."""

	target: Address


type DirectTransfer = DirectCall | TailJump


def direct_transfer(
	instruction: CsInsn, machine: Machine, own_start: Address, starts: frozenset[Address]
) -> DirectTransfer | None:
	callee = call_target(instruction, machine)
	if callee is not None:
		target = normalized(Address(callee), machine)
		return DirectCall(target=target) if target in starts else None
	branch = branch_target(instruction, machine)
	if branch is None:
		return None
	destination = normalized(Address(branch.target), machine)
	return (
		TailJump(target=destination) if destination in starts and destination != own_start else None
	)


def call_target(instruction: CsInsn, machine: Machine) -> int | None:
	match machine.family:
		case InstructionFamily.X86:
			return _x86_call_target(instruction)
		case InstructionFamily.ARM:
			return _arm_call_target(instruction)
		case _ as unreachable:
			assert_never(unreachable)


def _x86_call_target(instruction: CsInsn) -> int | None:
	if instruction.mnemonic != "call":
		return None
	operand = instruction.operands[0]
	if operand.type != x86_const.X86_OP_IMM:
		return None
	return operand.imm


def _arm_call_target(instruction: CsInsn) -> int | None:
	if arm_mnemonic(instruction) not in ARM_CALLS:
		return None
	operand = instruction.operands[0]
	if operand.type != arm_const.ARM_OP_IMM:
		return None
	return operand.imm


def _is_control_transfer(instruction: CsInsn, machine: Machine) -> bool:
	match machine.family:
		case InstructionFamily.X86:
			return instruction.mnemonic in X86_TRANSFERS
		case InstructionFamily.ARM:
			return arm_mnemonic(instruction) in ARM_TRANSFERS or (
				arm_mnemonic(instruction) in ("ldr", "mov", "add") and _writes_pc(instruction)
			)
		case _ as unreachable:
			assert_never(unreachable)


class RegisterSite(Struct):
	"""An indirect call or branch whose target is a register's value."""

	operand: CsOperand


class MemorySite(Struct):
	"""An indirect call or branch whose target is loaded from memory."""

	operand: CsOperand


def indirect_operand(instruction: CsInsn, machine: Machine) -> RegisterSite | MemorySite | None:
	if instruction.mnemonic == ".byte":
		return None
	if not instruction.operands:
		return None
	match machine.family:
		case InstructionFamily.X86:
			return _x86_indirect_operand(instruction)
		case InstructionFamily.ARM:
			return _arm_indirect_operand(instruction)
		case _ as unreachable:
			assert_never(unreachable)


def _x86_indirect_operand(instruction: CsInsn) -> RegisterSite | MemorySite | None:
	operand = instruction.operands[0]
	if instruction.mnemonic in ("call", "jmp"):
		if operand.type == x86_const.X86_OP_MEM:
			return MemorySite(operand=operand)
		if operand.type == x86_const.X86_OP_REG:
			return RegisterSite(operand=operand)
	return None


def _arm_indirect_operand(instruction: CsInsn) -> RegisterSite | MemorySite | None:
	operand = instruction.operands[0]
	base_mnemonic = arm_mnemonic(instruction)
	if base_mnemonic == "blx" and operand.type == arm_const.ARM_OP_REG:
		return RegisterSite(operand=operand)
	if base_mnemonic == "bx" and operand.reg not in (arm_const.ARM_REG_LR, arm_const.ARM_REG_PC):
		return RegisterSite(operand=operand)
	if base_mnemonic == "mov" and _writes_pc(instruction):
		source = instruction.operands[1]
		if source.type == arm_const.ARM_OP_REG and source.reg != arm_const.ARM_REG_LR:
			return RegisterSite(operand=source)
		return None
	if base_mnemonic == "ldr" and _writes_pc(instruction):
		memory = instruction.operands[1]
		if memory.mem.base != arm_const.ARM_REG_SP and memory.mem.index == 0:
			return MemorySite(operand=memory)
	return None


def _block_end_successors(
	instruction: CsInsn, machine: Machine, by_address: Mapping[int, int], next_address: int | None
) -> tuple[Address, ...]:
	branch = branch_target(instruction, machine)
	if branch is not None:
		return (
			*((Address(next_address),) if branch.conditional and next_address is not None else ()),
			*((Address(branch.target),) if branch.target in by_address else ()),
		)
	fallthrough = (Address(next_address),) if next_address is not None else ()
	if instruction.mnemonic == ".byte":
		return fallthrough
	return fallthrough if _transfer_returns(instruction, machine) else ()


def _transfer_returns(instruction: CsInsn, machine: Machine) -> bool:
	match machine.family:
		case InstructionFamily.X86:
			return instruction.mnemonic in X86_CALLS or is_returning_trap(instruction, machine)
		case InstructionFamily.ARM:
			return (
				arm_mnemonic(instruction) in ARM_CALLS
				or (
					arm_mnemonic(instruction) == "pop"
					and arm_const.ARM_REG_PC not in instruction.regs_access()[1]
				)
				or is_returning_trap(instruction, machine)
				or arm_predicated(instruction)
			)
		case _ as unreachable:
			assert_never(unreachable)


def is_returning_trap(instruction: CsInsn, machine: Machine) -> bool:
	match machine.family:
		case InstructionFamily.X86:
			return instruction.mnemonic in X86_RETURNING_TRAPS
		case InstructionFamily.ARM:
			return arm_mnemonic(instruction) in ARM_RETURNING_TRAPS
		case _ as unreachable:
			assert_never(unreachable)


def _block_ends(instruction: CsInsn, machine: Machine) -> bool:
	return (
		instruction.mnemonic == ".byte"
		or branch_target(instruction, machine) is not None
		or _is_control_transfer(instruction, machine)
	)


def _build_blocks(
	instructions: tuple[CsInsn, ...],
	machine: Machine,
	dispatches: Mapping[int, _JumpTable | None],
) -> tuple[Block, ...]:
	by_address = {instruction.address: index for index, instruction in enumerate(instructions)}
	branched_to = frozenset(
		branch.target
		for instruction in instructions
		if (branch := branch_target(instruction, machine)) is not None
	)
	entered = branched_to | {
		target for table in dispatches.values() if table is not None for target in table.targets
	}
	cases = {
		address: table.targets
		for address, table in dispatches.items()
		if table is not None
		and entered.isdisjoint(table.guarded)
		and all(target in by_address for target in table.targets)
	}
	targets = branched_to | {target for targets in cases.values() for target in targets}
	blocks: list[Block] = []
	current: list[CsInsn] = []
	for index, instruction in enumerate(instructions):
		if (
			instruction.address in targets
			or indirect_operand(instruction, machine) is not None
			or call_target(instruction, machine) is not None
			or is_returning_trap(instruction, machine)
			or (
				(branch := branch_target(instruction, machine)) is not None
				and branch.target not in by_address
			)
		) and current:
			blocks.append(
				Block(
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
				Block(
					start=Address(current[0].address),
					instructions=tuple(current),
					successors=cases[instruction.address]
					if instruction.address in cases
					else ()
					if instruction.address in dispatches
					else _block_end_successors(instruction, machine, by_address, next_address),
				)
			)
			current = []
	if current:
		blocks.append(
			Block(start=Address(current[0].address), instructions=tuple(current), successors=())
		)
	every_start = tuple(block.start for block in blocks)
	return tuple(
		Block(start=block.start, instructions=block.instructions, successors=every_start)
		if block.instructions[-1].address in dispatches
		and block.instructions[-1].address not in cases
		else block
		for block in blocks
	)
