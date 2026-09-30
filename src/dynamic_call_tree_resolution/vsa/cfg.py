# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Control-flow graphs of function code."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from capstone import Cs, arm_const, x86_const
from salix import Struct

from dynamic_call_tree_resolution.model import Address, Machine, aligned
from dynamic_call_tree_resolution.points_to import instruction_runs, memory_at
from dynamic_call_tree_resolution.vsa.abi import (
	ARM_CALLS,
	ARM_CONDITIONAL,
	ARM_RETURNING_TRAPS,
	ARM_TRANSFERS,
	X86_CALLS,
	X86_RETURNING_TRAPS,
	X86_TRANSFERS,
	arm_mnemonic,
	arm_predicated,
	normalized,
)

if TYPE_CHECKING:
	from collections.abc import Mapping

	from capstone import CsInsn, CsOperand

	from dynamic_call_tree_resolution.model import Function, Program


class Block(Struct):
	"""One basic block of a function's code."""

	start: Address
	instructions: tuple[CsInsn, ...]
	successors: tuple[Address, ...]


def _function_address(function: Function) -> Address:
	return function.address


def control_flow_graphs(
	program: Program, disassembler: Cs
) -> dict[Address, tuple[Function, tuple[Block, ...]]]:
	seen: set[Address] = set()
	blocks: dict[Address, tuple[Function, tuple[Block, ...]]] = {}
	for function in sorted(program.functions.values(), key=_function_address):
		# ARM symbol addresses carry the Thumb bit; strip it so the code
		# decodes from the aligned start, and skip the symbol/DWARF twin.
		start = aligned(function.address) if program.machine is Machine.EM_ARM else function.address
		if start in seen:
			continue
		runs = instruction_runs(program, start, function.size)
		if not any(code for _, code in runs):
			continue
		seen.add(start)
		decoded = tuple(
			_decoded(disassembler, program, address, code, (start, start + function.size))
			for address, code in runs
		)
		blocks[start] = (
			function,
			_build_blocks(
				tuple(instruction for run, _ in decoded for instruction in run),
				program.machine,
				{address: table for _, tables in decoded for address, table in tables.items()},
			),
		)
	return blocks


class _JumpTable(Struct):
	"""The case addresses a bounded dispatch's table holds."""

	targets: tuple[Address, ...]
	end: int
	"""The first address past the table, where decoding resumes."""
	guarded: tuple[int, ...]
	"""Instructions from the bound check's branch to the dispatch; entering one skips the check."""


def _decoded(
	disassembler: Cs, program: Program, start: int, code: bytes, function: tuple[int, int]
) -> tuple[tuple[CsInsn, ...], Mapping[int, _JumpTable | None]]:
	instructions: list[CsInsn] = []
	dispatches: dict[int, _JumpTable | None] = {}
	offset = 0
	while offset < len(code):
		for instruction in disassembler.disasm(code[offset:], start + offset):
			instructions.append(instruction)
			if program.machine is Machine.EM_ARM and _dispatches(instruction):
				table = _jump_table(instructions, program, function)
				dispatches[instruction.address] = table
				if table is not None:
					offset = table.end - start
					break
		else:
			break
	return tuple(instructions), dispatches


def _dispatches(instruction: CsInsn) -> bool:
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
	dispatch = instructions[-1]
	mnemonic = arm_mnemonic(dispatch)
	table_base = (
		None
		if mnemonic == "add"
		else dispatch.address + 4
		if mnemonic in ("tbb", "tbh")
		else _adr_target(instructions[-2])
		if len(instructions) >= 2
		else None
	)
	guard = len(instructions) - (2 if mnemonic in ("tbb", "tbh") else 3)
	if table_base is None or guard < 1:
		return None
	count = _bound(
		instructions[guard - 1],
		instructions[guard],
		dispatch.operands[-1].mem.index,
	)
	if count is None:
		return None
	size = {"tbb": 1, "tbh": 2}.get(mnemonic, 4)
	table = memory_at(program, Address(table_base), size * count)
	if len(table) != size * count:
		return None
	entries = tuple(
		int.from_bytes(table[index : index + size], "little")
		for index in range(0, len(table), size)
	)
	targets = tuple(
		Address(dispatch.address + 4 + 2 * entry if size < 4 else aligned(Address(entry)))
		for entry in entries
	)
	if not all(function[0] <= target < function[1] for target in targets):
		return None
	return _JumpTable(
		targets=targets,
		end=table_base + size * count + (size * count) % 2,
		guarded=tuple(instruction.address for instruction in instructions[guard:]),
	)


def _adr_target(instruction: CsInsn) -> int | None:
	if (
		arm_mnemonic(instruction) not in ("add", "adr")
		or len(instruction.operands) < 2
		or instruction.operands[-1].type != arm_const.ARM_OP_IMM
		or (
			arm_mnemonic(instruction) == "add"
			and instruction.operands[1].reg != arm_const.ARM_REG_PC
		)
	):
		return None
	return ((instruction.address + 4) & ~3) + instruction.operands[-1].imm


def _bound(compare: CsInsn, branch: CsInsn, index: int) -> int | None:
	if (
		arm_mnemonic(compare) != "cmp"
		or compare.operands[0].reg != index
		or compare.operands[1].type != arm_const.ARM_OP_IMM
		or arm_mnemonic(branch) != "b"
		or branch.cc != arm_const.ARM_CC_HI
	):
		return None
	return compare.operands[1].imm + 1


class _BranchTarget(Struct):
	target: int
	conditional: bool


def branch_target(instruction: CsInsn, machine: Machine) -> _BranchTarget | None:
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
	if machine.is_x86:
		if instruction.mnemonic != "call":
			return None
		operand = instruction.operands[0]
		if operand.type != x86_const.X86_OP_IMM:
			return None
		return operand.imm
	if arm_mnemonic(instruction) not in ARM_CALLS:
		return None
	operand = instruction.operands[0]
	if operand.type != arm_const.ARM_OP_IMM:
		return None
	return operand.imm


def _is_control_transfer(instruction: CsInsn, machine: Machine) -> bool:
	if machine.is_x86:
		return instruction.mnemonic in X86_TRANSFERS
	return arm_mnemonic(instruction) in ARM_TRANSFERS or (
		arm_mnemonic(instruction) in ("ldr", "mov", "add") and _writes_pc(instruction)
	)


def indirect_operand(
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
	base_mnemonic = arm_mnemonic(instruction)
	if base_mnemonic == "blx" and operand.type == arm_const.ARM_OP_REG:
		return "register", operand
	if base_mnemonic == "bx" and operand.reg not in (arm_const.ARM_REG_LR, arm_const.ARM_REG_PC):
		return "register", operand
	if base_mnemonic == "mov" and _writes_pc(instruction):
		source = instruction.operands[1]
		if source.type == arm_const.ARM_OP_REG and source.reg != arm_const.ARM_REG_LR:
			return "register", source
		return None
	if base_mnemonic == "ldr" and _writes_pc(instruction):
		memory = instruction.operands[1]
		if memory.mem.base != arm_const.ARM_REG_SP and memory.mem.index == 0:
			return "memory", memory
	return None


def _successors(
	instruction: CsInsn, machine: Machine, by_address: Mapping[int, int], next_address: int | None
) -> tuple[Address, ...]:
	branch = branch_target(instruction, machine)
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
		if machine.is_x86:
			return (
				fallthrough
				if instruction.mnemonic in X86_CALLS or is_returning_trap(instruction, machine)
				else ()
			)
		if (
			arm_mnemonic(instruction) in ARM_CALLS
			or (
				arm_mnemonic(instruction) == "pop"
				and arm_const.ARM_REG_PC not in instruction.regs_access()[1]
			)
			or is_returning_trap(instruction, machine)
			or arm_predicated(instruction)
		):
			return fallthrough
		return ()
	raise AssertionError  # every block-ending instruction matches an arm above


def is_returning_trap(instruction: CsInsn, machine: Machine) -> bool:
	return (
		instruction.mnemonic in X86_RETURNING_TRAPS
		if machine.is_x86
		else arm_mnemonic(instruction) in ARM_RETURNING_TRAPS
	)


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
		if table is not None and entered.isdisjoint(table.guarded)
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
					successors=tuple(
						target for target in cases[instruction.address] if target in by_address
					)
					if instruction.address in cases
					else ()
					if instruction.address in dispatches
					else _successors(instruction, machine, by_address, next_address),
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
