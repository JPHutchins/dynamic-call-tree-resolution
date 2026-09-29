# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Thumb direct-branch targets, from capstone to the VSA's CFG."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest
from capstone import CS_ARCH_ARM, CS_MODE_THUMB, Cs

from dynamic_call_tree_resolution import assignments, call_site_candidates, extract_call_sites, load
from dynamic_call_tree_resolution.model import aligned
from dynamic_call_tree_resolution.points_to import memory_at

if TYPE_CHECKING:
	from capstone import CsInsn

pytestmark = pytest.mark.image

FIXTURES = Path(__file__).parent / "fixtures"
ARM_ELFS = (
	FIXTURES / "hello_zephyr_qemu_cortex_m3.elf",
	FIXTURES / "sensor-two-impl" / "zephyr" / "zephyr.elf",
)

type Encoding = Literal["B T1", "B T2", "CBZ T1", "B T3", "B T4", "BL T1"]


def _sign_extended(value: int, bits: int) -> int:
	return value - (1 << bits) if value >> (bits - 1) else value


def _narrow(halfword: int) -> tuple[Encoding, int] | None:
	if halfword >> 12 == 0b1101 and halfword >> 8 & 0xF < 0b1110:
		return "B T1", _sign_extended((halfword & 0xFF) << 1, 9)
	if halfword >> 11 == 0b11100:
		return "B T2", _sign_extended((halfword & 0x7FF) << 1, 12)
	if halfword & 0xF500 == 0xB100:
		return "CBZ T1", (halfword >> 9 & 1) << 6 | (halfword >> 3 & 0x1F) << 1
	return None


def _wide(first: int, second: int) -> tuple[Encoding, int] | None:
	s, j1, j2 = first >> 10 & 1, second >> 13 & 1, second >> 11 & 1
	far = (
		(1 - (j1 ^ s)) << 23 | (1 - (j2 ^ s)) << 22 | (first & 0x3FF) << 12 | (second & 0x7FF) << 1
	)
	match second >> 14 & 1, second >> 12 & 1:
		case 0, 0 if first >> 6 & 0xF < 0b1110:
			near = j2 << 19 | j1 << 18 | (first & 0x3F) << 12 | (second & 0x7FF) << 1
			return "B T3", _sign_extended(s << 20 | near, 21)
		case 0, 1:
			return "B T4", _sign_extended(s << 24 | far, 25)
		case 1, 1:
			return "BL T1", _sign_extended(s << 24 | far, 25)
		case _:
			return None


def _encoded_branch(instruction: CsInsn) -> tuple[Encoding, int] | None:
	halfwords = [
		int.from_bytes(instruction.bytes[offset : offset + 2], "little")
		for offset in range(0, instruction.size, 2)
	]
	match halfwords:
		case [halfword]:
			encoded = _narrow(halfword)
		case [first, second] if first >> 11 == 0b11110 and second >> 15:
			encoded = _wide(first, second)
		case _:
			encoded = None
	return None if encoded is None else (encoded[0], instruction.address + 4 + encoded[1])


def _instructions(elf: Path) -> list[CsInsn]:
	program = load(elf)
	disassembler = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
	disassembler.detail = True
	return [
		instruction
		for start, size in sorted(
			{(aligned(function.address), function.size) for function in program.functions.values()}
		)
		for instruction in disassembler.disasm(memory_at(program, start, size), start)
	]


def test_capstone_reports_every_direct_branch_target_absolute() -> None:
	branches = [
		(instruction, encoded)
		for elf in ARM_ELFS
		for instruction in _instructions(elf)
		if (encoded := _encoded_branch(instruction)) is not None
	]
	assert {encoding for _, (encoding, _) in branches} == {
		"B T1",
		"B T2",
		"CBZ T1",
		"B T3",
		"B T4",
		"BL T1",
	}
	assert [
		(instruction.address, instruction.mnemonic)
		for instruction, (_, target) in branches
		if instruction.operands[-1].imm != target
	] == []


def test_hello_z_cstart_loop_site_has_no_candidates() -> None:
	program = load(ARM_ELFS[0])
	resolved = {assignment.slot: assignment for assignment in assignments(program)}
	(site,) = (site for site in extract_call_sites(program) if site.site_address == 0xEB4)
	assert call_site_candidates(program, site, resolved) == frozenset()
