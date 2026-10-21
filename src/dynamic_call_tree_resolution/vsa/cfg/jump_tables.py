# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Bounding a dispatch through a jump table to its cases."""

from __future__ import annotations

from functools import partial, reduce
from itertools import takewhile
from typing import TYPE_CHECKING, Final, assert_never

from capstone import arm_const
from salix import Struct

from dynamic_call_tree_resolution.model import Address, InstructionFamily, Machine, aligned
from dynamic_call_tree_resolution.span import Span
from dynamic_call_tree_resolution.vsa.abi import (
	ARM_LOAD_WIDTHS,
	arm_predicated,
)
from dynamic_call_tree_resolution.vsa.cfg.operands import (
	WORD_MASK,
	arm_operands,
	memory_operand_address,
	memory_operand_base,
	operand_sum,
	operand_value,
	pc_relative,
	read_memory,
)
from dynamic_call_tree_resolution.vsa.cfg.transfers import (
	block_ends,
	writes_pc,
)

if TYPE_CHECKING:
	from collections.abc import Callable, Mapping

	from capstone import CsInsn

	from dynamic_call_tree_resolution.model import Program


_WINDOW: Final = 16


_MAX_CASES: Final = 1024


_NOTHING_KNOWN: Final[Mapping[int, int]] = {}


class JumpTable(Struct):
	"""The case addresses a bounded dispatch reaches."""

	targets: tuple[Address, ...]
	skip: int | None
	"""The end of a table that follows the dispatch, which the decoder skips."""
	guarded: tuple[int, ...]
	"""Instructions no branch may enter without skipping the bound or a value the dispatch reads."""


class _Case(Struct):
	"""Where a dispatch goes for one index."""

	target: Address
	table: Span | None
	"""The bytes the dispatch read the target from."""


def dispatches_indirectly(instruction: CsInsn, machine: Machine) -> bool:
	match machine.family:
		case InstructionFamily.X86:
			return False
		case InstructionFamily.ARM:
			return _arm_dispatches(instruction)
		case _ as unreachable:
			assert_never(unreachable)


def _arm_dispatches(instruction: CsInsn) -> bool:
	match instruction.id:
		case arm_const.ARM_INS_TBB | arm_const.ARM_INS_TBH:
			return True
		case arm_const.ARM_INS_LDR:
			return writes_pc(instruction) and instruction.operands[1].mem.index != 0
		case arm_const.ARM_INS_ADD:
			return writes_pc(instruction)
		case _:
			return False


def jump_table(instructions: list[CsInsn], program: Program, function: Span) -> JumpTable | None:
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
			if block_ends(instructions[index], Machine.EM_ARM)
		),
		floor,
	)


def _bounds(compare: CsInsn, branch: CsInsn) -> bool:
	return (
		compare.id == arm_const.ARM_INS_CMP
		and not arm_predicated(compare)
		and compare.operands[1].type == arm_const.ARM_OP_IMM
		and branch.id == arm_const.ARM_INS_B
		and branch.cc == arm_const.ARM_CC_HI
	)


def _bounded(
	instructions: list[CsInsn],
	start: int,
	guard: int,
	program: Program,
	function: Span,
) -> JumpTable | None:
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
	if len(cases) <= compare.operands[1].imm or not all(case.target in function for case in cases):
		return None
	tables = tuple(case.table for case in cases if case.table is not None)
	end = max((table.end for table in tables), default=0)
	return JumpTable(
		targets=tuple(case.target for case in cases),
		skip=end + end % 2
		if dispatch.address + dispatch.size
		<= min((table.start for table in tables), default=-1)
		<= dispatch.address + dispatch.size + 2
		else None,
		guarded=tuple(instruction.address for instruction in instructions[start + 1 :]),
	)


def _is_case(case: _Case | None) -> bool:
	return case is not None


def _case(program: Program, dispatch: CsInsn, registers: Mapping[int, int]) -> _Case | None:
	match dispatch.id, arm_operands(dispatch):
		case (arm_const.ARM_INS_TBB | arm_const.ARM_INS_TBH) as table_branch, [table]:
			return _read_case(
				program,
				memory_operand_address(registers, table, dispatch.address + 4),
				ARM_LOAD_WIDTHS[
					arm_const.ARM_INS_LDRB
					if table_branch == arm_const.ARM_INS_TBB
					else arm_const.ARM_INS_LDRH
				],
				partial(_halfwords_past, dispatch.address + 4),
			)
		case arm_const.ARM_INS_LDR, [_, table]:
			return _read_case(
				program,
				memory_operand_address(
					registers, table, memory_operand_base(program, dispatch, registers, table)
				),
				ARM_LOAD_WIDTHS[arm_const.ARM_INS_LDR],
				_absolute_target,
			)
		case arm_const.ARM_INS_ADD, [_, base, offset] if (
			base.reg == arm_const.ARM_REG_PC
			and (value := operand_value(registers, offset)) is not None
		):
			return _Case(target=Address(pc_relative(program, dispatch) + value), table=None)
		case _:
			return None


def _read_case(
	program: Program,
	address: int | None,
	width: int,
	target: Callable[[int], Address],
) -> _Case | None:
	entry = None if address is None else read_memory(program, address, width)
	return (
		None
		if address is None or entry is None
		else _Case(
			target=target(entry), table=Span(start=Address(address), end=Address(address + width))
		)
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
	return kept if value is None else {**kept, instruction.operands[0].reg: value & WORD_MASK}


def _computed(program: Program, registers: Mapping[int, int], instruction: CsInsn) -> int | None:
	match instruction.id, arm_operands(instruction):
		case (arm_const.ARM_INS_LDR | arm_const.ARM_INS_LDRB | arm_const.ARM_INS_LDRH) as load, [
			_,
			source,
		]:
			address = memory_operand_address(
				registers, source, memory_operand_base(program, instruction, registers, source)
			)
			return None if address is None else read_memory(program, address, ARM_LOAD_WIDTHS[load])
		case arm_const.ARM_INS_ADR, [_, offset]:
			return pc_relative(program, instruction) + offset.imm
		case ((arm_const.ARM_INS_ADD | arm_const.ARM_INS_ADDW), [_, base, offset]) if (
			base.reg == arm_const.ARM_REG_PC and offset.type == arm_const.ARM_OP_IMM
		):
			return pc_relative(program, instruction) + offset.imm
		case ((arm_const.ARM_INS_ADD | arm_const.ARM_INS_ADDW), [_, augend, addend]):
			return operand_sum(registers, augend, addend)
		case (
			(
				arm_const.ARM_INS_MOV
				| arm_const.ARM_INS_MOVS
				| arm_const.ARM_INS_MOVW
				| arm_const.ARM_INS_LSL
			),
			[_, source],
		):
			return operand_value(registers, source)
		case arm_const.ARM_INS_MOVT, [destination, high] if (
			low := registers.get(destination.reg)
		) is not None:
			return (high.imm << 16) | (low & 0xFFFF)
		case _:
			return None
