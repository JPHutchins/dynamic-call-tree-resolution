# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Where an instruction transfers control, and through which operand."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from capstone import arm_const, x86_const
from salix import Struct

from dynamic_call_tree_resolution.model import Address, InstructionFamily, Machine
from dynamic_call_tree_resolution.vsa.abi import (
	ARM_CALLS,
	ARM_RETURNING_TRAPS,
	ARM_TRANSFERS,
	X86_RETURNING_TRAPS,
	X86_TRANSFERS,
	arm_predicated,
	is_data,
	normalized,
)

if TYPE_CHECKING:
	from capstone import CsInsn, CsOperand


class BranchTarget(Struct):
	"""Where a branch goes, and whether it can fall through instead."""

	target: int
	conditional: bool


def branch_target(instruction: CsInsn, machine: Machine) -> BranchTarget | None:
	match machine.family:
		case InstructionFamily.X86:
			return _x86_branch_target(instruction)
		case InstructionFamily.ARM:
			return _arm_branch_target(instruction)
		case _ as unreachable:
			assert_never(unreachable)


def _x86_branch_target(instruction: CsInsn) -> BranchTarget | None:
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
	return BranchTarget(target=operand.imm, conditional=conditional)


def _arm_branch_target(instruction: CsInsn) -> BranchTarget | None:
	match instruction.id:
		case arm_const.ARM_INS_B:
			return _immediate_target(
				instruction.operands[0], conditional=arm_predicated(instruction)
			)
		case arm_const.ARM_INS_CBZ | arm_const.ARM_INS_CBNZ:
			return _immediate_target(instruction.operands[1], conditional=True)
		case _:
			return None


def _immediate_target(operand: CsOperand, *, conditional: bool) -> BranchTarget | None:
	if operand.type != arm_const.ARM_OP_IMM:
		return None  # pragma: no cover
	return BranchTarget(target=operand.imm, conditional=conditional)


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
	if instruction.id not in ARM_CALLS:
		return None
	operand = instruction.operands[0]
	if operand.type != arm_const.ARM_OP_IMM:
		return None
	return operand.imm


def writes_pc(instruction: CsInsn) -> bool:
	return (
		bool(instruction.operands)
		and instruction.operands[0].type == arm_const.ARM_OP_REG
		and instruction.operands[0].reg == arm_const.ARM_REG_PC
	)


def is_control_transfer(instruction: CsInsn, machine: Machine) -> bool:
	match machine.family:
		case InstructionFamily.X86:
			return instruction.mnemonic in X86_TRANSFERS
		case InstructionFamily.ARM:
			return instruction.id in ARM_TRANSFERS or (
				instruction.id
				in (arm_const.ARM_INS_LDR, arm_const.ARM_INS_MOV, arm_const.ARM_INS_ADD)
				and writes_pc(instruction)
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
	if is_data(instruction):
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
	match instruction.id:
		case arm_const.ARM_INS_BLX if operand.type == arm_const.ARM_OP_REG:
			return RegisterSite(operand=operand)
		case arm_const.ARM_INS_BX if operand.reg not in (
			arm_const.ARM_REG_LR,
			arm_const.ARM_REG_PC,
		):
			return RegisterSite(operand=operand)
		case arm_const.ARM_INS_MOV if writes_pc(instruction):
			source = instruction.operands[1]
			if source.type == arm_const.ARM_OP_REG and source.reg != arm_const.ARM_REG_LR:
				return RegisterSite(operand=source)
			return None
		case arm_const.ARM_INS_LDR if writes_pc(instruction):
			memory = instruction.operands[1]
			if memory.mem.base != arm_const.ARM_REG_SP and memory.mem.index == 0:
				return MemorySite(operand=memory)
			return None
		case _:
			return None


def is_returning_trap(instruction: CsInsn, machine: Machine) -> bool:
	match machine.family:
		case InstructionFamily.X86:
			return instruction.mnemonic in X86_RETURNING_TRAPS
		case InstructionFamily.ARM:
			return instruction.id in ARM_RETURNING_TRAPS
		case _ as unreachable:
			assert_never(unreachable)


def arm_falls_through(instruction: CsInsn, machine: Machine) -> bool:
	return (
		instruction.id in ARM_CALLS
		or (
			instruction.id == arm_const.ARM_INS_POP
			and arm_const.ARM_REG_PC not in instruction.regs_access()[1]
		)
		or is_returning_trap(instruction, machine)
	)


def block_ends(instruction: CsInsn, machine: Machine) -> bool:
	return (
		is_data(instruction)
		or branch_target(instruction, machine) is not None
		or is_control_transfer(instruction, machine)
	)
