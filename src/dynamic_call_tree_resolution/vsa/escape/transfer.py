# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""What one instruction lets escape."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, assert_never

from capstone import CS_AC_READ, arm_const
from salix import Struct

from dynamic_call_tree_resolution.model import Address
from dynamic_call_tree_resolution.vsa.abi import (
	ARM_ADDITION_SIGNS,
	ARM_ARGUMENT_REGISTERS,
	ARM_CALLER_SAVED,
	ARM_RETURNING_TRAPS,
	ARM_UNTRACKED_STORES,
	arm_predicated,
	is_data,
	normalized,
)
from dynamic_call_tree_resolution.vsa.cfg import branch_target, call_target
from dynamic_call_tree_resolution.vsa.escape.derive import (
	LISTED_BASES,
	MOVES,
	PLACED_STORES,
	derived_destination,
	literal_address,
	operands_of,
	store_target,
	written_back,
)
from dynamic_call_tree_resolution.vsa.escape.image import (
	Fill,
	Image,
	overlapping,
	reproduced,
)
from dynamic_call_tree_resolution.vsa.escape.values import (
	NOTHING,
	Held,
	Registers,
	size,
	union,
	unknown_offset,
)

if TYPE_CHECKING:
	from collections.abc import Mapping

	from capstone import ArmCsOperand, CsInsn

	from dynamic_call_tree_resolution.vsa.escape.image import Span


_COMPARES: Final = frozenset(
	{
		arm_const.ARM_INS_CMP,
		arm_const.ARM_INS_CMN,
		arm_const.ARM_INS_TST,
		arm_const.ARM_INS_TEQ,
		arm_const.ARM_INS_CBZ,
		arm_const.ARM_INS_CBNZ,
	}
)


_DERIVATIONS: Final = MOVES | {arm_const.ARM_INS_MOVT} | ARM_ADDITION_SIGNS.keys()


_DISPATCHES: Final = frozenset({arm_const.ARM_INS_TBB, arm_const.ARM_INS_TBH})


RETURNED: Final = frozenset({arm_const.ARM_REG_R0, arm_const.ARM_REG_R1})


class _Call(Struct):
	target: Address | None


class _Return(Struct):
	passed: frozenset[int]
	"""The registers that carry the return value."""


class _Jump(Struct):
	"""A write to the program counter that neither calls nor returns."""

	resolved: bool
	"""The control flow graph knows where it goes."""


type _Control = _Call | _Return | _Jump


class Scope(Struct):
	"""What the scan of one function needs to know about it."""

	blocks: frozenset[Address]
	returned: frozenset[int]
	"""The registers its return value occupies; none when DWARF says it returns ``void``."""


class Effects(Struct):
	"""What one instruction lets escape, overwrites, or stores through."""

	escaped: Held = NOTHING
	filled: Held = NOTHING
	"""The destination of a copy or zero fill whose arguments aren't constant."""
	overwritten: frozenset[Span] = frozenset()
	"""Objects an initialization call leaves without their image value."""
	site: Address | None = None
	"""The store instruction's address."""
	store: Held = NOTHING
	"""Where the store writes."""
	literal: int | None = None
	"""The address of the word a PC-relative load reads."""


_NO_EFFECTS: Final = Effects()


def step(
	image: Image,
	scope: Scope,
	instruction: CsInsn,
	registers: Registers,
	successors: tuple[Address, ...],
) -> tuple[Registers, Effects]:
	if is_data(instruction) or instruction.id == arm_const.ARM_INS_IT:
		return registers, _NO_EFFECTS
	operands = operands_of(instruction)
	written = frozenset(instruction.regs_access()[1])
	control = _control(image, scope, instruction, operands, written, successors)
	fill = _filled(image, control, registers.constants)
	after = _after(
		instruction,
		registers,
		{
			**derived_destination(image, instruction, operands, registers),
			**written_back(instruction, operands, registers),
		},
		written | (frozenset(ARM_CALLER_SAVED) if isinstance(control, _Call) else frozenset[int]()),
	)
	return after, Effects(
		escaped=union(
			*(
				registers.held.get(register, NOTHING)
				for register in _escaping(image, instruction, operands, control, fill, registers)
			)
		),
		filled=_walked(image, control, fill, registers),
		overwritten=frozenset() if fill is None else fill,
		site=Address(instruction.address) if instruction.id in PLACED_STORES else None,
		store=store_target(instruction, operands, registers),
		literal=literal_address(image.program, instruction, operands),
	)


def _after(
	instruction: CsInsn,
	registers: Registers,
	new: Mapping[int, tuple[Held, int | None]],
	written: frozenset[int],
) -> Registers:
	"""The registers after the instruction; a predicated one may not run, so it only adds."""
	predicated = arm_predicated(instruction)
	kept = {
		register: held
		for register, held in registers.held.items()
		if predicated or register not in written
	}
	return Registers(
		held={
			**kept,
			**{
				register: value
				for register, value in (
					(register, union(kept.get(register, NOTHING), value) if predicated else value)
					for register, (value, _) in new.items()
				)
				if size(value)
			},
		},
		constants={
			**{
				register: value
				for register, value in registers.constants.items()
				if register not in written
				or (predicated and register in new and new[register][1] == value)
			},
			**(
				{}
				if predicated
				else {register: value for register, (_, value) in new.items() if value is not None}
			),
		},
	)


def _control(
	image: Image,
	scope: Scope,
	instruction: CsInsn,
	operands: tuple[ArmCsOperand, ...],
	written: frozenset[int],
	successors: tuple[Address, ...],
) -> _Control | None:
	machine = image.program.machine
	called = call_target(instruction, machine)
	if called is not None:
		return _Call(target=normalized(Address(called), machine))
	if instruction.id in ARM_RETURNING_TRAPS:
		return _Call(target=None)
	branch = branch_target(instruction, machine)
	if branch is not None:
		return (
			None
			if normalized(Address(branch.target), machine) in scope.blocks
			else _Call(target=normalized(Address(branch.target), machine))
		)
	if instruction.id in (arm_const.ARM_INS_BX, arm_const.ARM_INS_BLX):
		return (
			_Return(passed=scope.returned)
			if instruction.id == arm_const.ARM_INS_BX and operands[0].reg == arm_const.ARM_REG_LR
			else _Call(target=None)
		)
	if instruction.id in _DISPATCHES or arm_const.ARM_REG_PC in written:
		return (
			_Return(passed=scope.returned)
			if _returns(instruction, operands)
			else _Jump(resolved=len(successors) > (1 if arm_predicated(instruction) else 0))
		)
	return None


def _returns(instruction: CsInsn, operands: tuple[ArmCsOperand, ...]) -> bool:
	"""Whether a write to the program counter takes it from the stack or the link register."""
	return (
		instruction.id == arm_const.ARM_INS_POP
		or any(
			operand.type == arm_const.ARM_OP_MEM and operand.mem.base == arm_const.ARM_REG_SP
			for operand in operands
		)
		or (instruction.id in LISTED_BASES and operands[0].reg == arm_const.ARM_REG_SP)
		or (
			instruction.id in MOVES
			and operands[1].type == arm_const.ARM_OP_REG
			and operands[1].reg == arm_const.ARM_REG_LR
		)
	)


def _filled(
	image: Image, control: _Control | None, constants: Mapping[int, int]
) -> frozenset[Span] | None:
	"""The objects a fill with constant arguments leaves without their image value, if it is one."""
	match (
		image.fills.get(control.target)
		if isinstance(control, _Call) and control.target is not None
		else None,
		constants.get(arm_const.ARM_REG_R0),
		constants.get(arm_const.ARM_REG_R1),
		constants.get(arm_const.ARM_REG_R2),
	):
		case Fill() as fill, int() as destination, int() as argument, int() as length:
			return frozenset(
				span
				for span in overlapping(image, destination, destination + length)
				if not reproduced(image, fill, span, destination, argument)
			)
		case _:
			return None


def _walked(
	image: Image, control: _Control | None, fill: frozenset[Span] | None, registers: Registers
) -> Held:
	"""The destination of a fill whose arguments aren't constant, which it writes as a range."""
	return (
		unknown_offset(registers.held.get(ARM_ARGUMENT_REGISTERS[0], NOTHING))
		if fill is None
		and isinstance(control, _Call)
		and control.target is not None
		and control.target in image.fills
		else NOTHING
	)


def _escaping(
	image: Image,
	instruction: CsInsn,
	operands: tuple[ArmCsOperand, ...],
	control: _Control | None,
	fill: frozenset[Span] | None,
	registers: Registers,
) -> frozenset[int]:
	"""The registers whose addresses the instruction puts somewhere the scan doesn't follow."""
	listed = frozenset({operands[0].reg}) if instruction.id in LISTED_BASES else frozenset[int]()
	memory = tuple(operand.mem for operand in operands if operand.type == arm_const.ARM_OP_MEM)
	return (
		frozenset(access.index for access in memory if access.index)
		| (
			frozenset(access.base for access in memory) | listed
			if instruction.id in ARM_UNTRACKED_STORES
			else frozenset[int]()
		)
		| (
			frozenset(
				operand.reg
				for operand in operands
				if operand.type == arm_const.ARM_OP_REG and operand.access & CS_AC_READ
			)
			- _consumed(instruction, operands, listed)
		)
		| _passed(image, control, fill, registers)
	)


def _consumed(
	instruction: CsInsn, operands: tuple[ArmCsOperand, ...], listed: frozenset[int]
) -> frozenset[int]:
	"""The registers the instruction reads only to compare or to derive another address."""
	return (
		frozenset(operand.reg for operand in operands if operand.type == arm_const.ARM_OP_REG)
		if instruction.id in _COMPARES
		or (
			instruction.id in _DERIVATIONS
			and operands[0].reg not in (arm_const.ARM_REG_SP, arm_const.ARM_REG_PC)
		)
		else listed
	)


def _passed(
	image: Image, control: _Control | None, fill: frozenset[Span] | None, registers: Registers
) -> frozenset[int]:
	match control:
		case None:
			return frozenset()
		case _Call(target=target):
			return (
				frozenset()
				if fill is not None or target in image.parameterless
				else frozenset(ARM_ARGUMENT_REGISTERS)
			)
		case _Return(passed=passed):
			return passed
		case _Jump(resolved=resolved):
			return frozenset() if resolved else frozenset(registers.held)
		case _ as unreachable:
			assert_never(unreachable)
