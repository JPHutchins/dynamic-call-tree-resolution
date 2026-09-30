# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Control-flow graphs of function code."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from capstone import Cs, arm_const, x86_const
from salix import Struct

from dynamic_call_tree_resolution.model import Address, Machine, aligned
from dynamic_call_tree_resolution.points_to import memory_at
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
		code = memory_at(program, start, function.size)
		if not code:
			continue
		seen.add(start)
		instructions = tuple(disassembler.disasm(code, start))
		blocks[start] = (function, _build_blocks(instructions, program.machine))
	return blocks


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
	return arm_mnemonic(instruction) in ARM_TRANSFERS


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


def _build_blocks(instructions: tuple[CsInsn, ...], machine: Machine) -> tuple[Block, ...]:
	by_address = {instruction.address: index for index, instruction in enumerate(instructions)}
	targets = frozenset(
		branch.target
		for instruction in instructions
		if (branch := branch_target(instruction, machine)) is not None
	)
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
					successors=_successors(instruction, machine, by_address, next_address),
				)
			)
			current = []
	if current:
		blocks.append(
			Block(start=Address(current[0].address), instructions=tuple(current), successors=())
		)
	return tuple(blocks)
