# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Arm instruction classes, keyed by capstone instruction id."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dynamic_call_tree_resolution.model import InstructionSet, Machine
from dynamic_call_tree_resolution.vsa.abi import arm_predicated, disassemblers, is_data
from dynamic_call_tree_resolution.vsa.cfg import (
	branch_target,
	call_target,
	indirect_operand,
	is_returning_trap,
)

if TYPE_CHECKING:
	from capstone import CsInsn


def _thumb(code: str) -> tuple[CsInsn, ...]:
	return tuple(disassemblers(None)[InstructionSet.T32].disasm(bytes.fromhex(code), 0x100))


def test_an_it_instruction_is_not_predicated_though_it_carries_its_first_condition() -> None:
	assert tuple(
		(instruction.mnemonic, arm_predicated(instruction))
		for instruction in _thumb("0cbf01200220")
	) == (("ite", False), ("moveq", True), ("movne", True))


def test_skipped_data_is_in_no_instruction_class() -> None:
	(data,) = _thumb("ffff")
	assert (
		is_data(data),
		arm_predicated(data),
		call_target(data, Machine.EM_ARM),
		branch_target(data, Machine.EM_ARM),
		indirect_operand(data, Machine.EM_ARM),
		is_returning_trap(data, Machine.EM_ARM),
	) == (True, False, None, None, None, False)
