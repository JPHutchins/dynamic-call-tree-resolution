# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The analysis decodes the Arm Zephyr fixtures as GNU objdump does (#75, item 3)."""

from __future__ import annotations

import re
import subprocess
from typing import TYPE_CHECKING, Final, assert_never

import pytest
from salix import Struct

from dynamic_call_tree_resolution import Address, load, resolve
from dynamic_call_tree_resolution.rtos import RtosChoice, rtos_model
from dynamic_call_tree_resolution.vsa.cfg import branch_target, call_target, control_flow_graphs

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path

	from capstone import CsInsn

	from dynamic_call_tree_resolution.model import Machine, Program
	from dynamic_call_tree_resolution.vsa.cfg import Block

pytestmark = pytest.mark.image

IMAGES: Final = ["hello", "sensor-threads", "sensor-two-impl", "synchronization"]
_LINE: Final = re.compile(r"^\s*([0-9a-f]+):\s+(?:[0-9a-f]{4,8} ?)+\s+(\S+)\s*([^;@]*)")
_CONDITION: Final = r"(?:eq|ne|cs|cc|hs|lo|mi|pl|vs|vc|hi|ls|ge|lt|gt|le)?(?:\.[nw])?"
_DIRECT: Final = re.compile(rf"(?:bl|blx|b{_CONDITION}|cbn?z)(?:\.[nw])?")
_REGISTER_BRANCH: Final = re.compile(rf"(?:blx|bx){_CONDITION}")
_REGISTER: Final = re.compile(r"r\d+|ip|sb|sl|fp")
_TARGET: Final = re.compile(r"([0-9a-f]+) <")
_PC_WRITE: Final = re.compile(r"(?:tbb|tbh|pop|ldm)\S*\s.*\bpc\b|\S+\s+pc\b")
_RETURN: Final = re.compile(r"(?:pop|ldm)\S*\s.*\bpc\}|ldr\S*\s+pc, \[sp\], #4")
_TABLE: Final = re.compile(r"(?:tbb|tbh)\S*\s|ldr\S*\s+pc, \[\w+, \w+, lsl #2\]")


class Listed(Struct):
	"""One instruction as objdump lists it."""

	text: str
	target: Address | None
	"""Where a direct branch or call goes."""


class Decoded(Struct):
	"""One fixture as objdump lists it and as the analysis decodes it."""

	program: Program
	listing: Mapping[Address, Listed]
	blocks: tuple[Block, ...]
	sites: frozenset[Address]


def _listing(elf: Path) -> Mapping[Address, Listed]:
	return {
		Address(int(address, 16)): Listed(
			text=f"{mnemonic} {operands.strip()}",
			target=_listed_target(mnemonic, line),
		)
		for line in subprocess.run(
			["arm-none-eabi-objdump", "-d", str(elf)], check=True, capture_output=True, text=True
		).stdout.splitlines()
		for match in (_LINE.match(line),)
		if match is not None
		for address, mnemonic, operands in (match.groups(""),)
		if not mnemonic.startswith(".")
	}


def _listed_target(mnemonic: str, line: str) -> Address | None:
	match _TARGET.search(line) if _DIRECT.fullmatch(mnemonic) else None:
		case None:
			return None
		case re.Match() as target:
			return Address(int(target.group(1), 16))
		case _ as unreachable:
			assert_never(unreachable)


def _decoded(elf: Path) -> Decoded:
	program = load(elf)
	return Decoded(
		program=program,
		listing=_listing(elf),
		blocks=tuple(
			block for _, blocks in control_flow_graphs(program).values() for block in blocks
		),
		sites=frozenset(
			site.site_address
			for site in resolve(program, rtos_model(program, RtosChoice.AUTO)).sites
		),
	)


@pytest.fixture(scope="module")
def decoded(zephyr_fixtures: Path) -> Mapping[str, Decoded]:
	return {name: _decoded(zephyr_fixtures / name / "zephyr" / "zephyr.elf") for name in IMAGES}


def _direct_target(instruction: CsInsn, machine: Machine) -> int | None:
	return next(
		(
			target
			for target in (
				call_target(instruction, machine),
				*(
					branch.target
					for branch in (branch_target(instruction, machine),)
					if branch is not None
				),
			)
			if target is not None
		),
		None,
	)


@pytest.mark.parametrize("name", IMAGES)
def test_every_decoded_instruction_starts_where_objdump_starts_one(
	decoded: Mapping[str, Decoded], name: str
) -> None:
	assert [
		f"{instruction.address:#x}"
		for block in decoded[name].blocks
		for instruction in block.instructions
		if instruction.address not in decoded[name].listing
	] == []


@pytest.mark.parametrize("name", IMAGES)
def test_no_decoded_instruction_is_bytes_the_decoder_could_not_read(
	decoded: Mapping[str, Decoded], name: str
) -> None:
	assert [
		f"{instruction.address:#x}"
		for block in decoded[name].blocks
		for instruction in block.instructions
		if instruction.id == 0
	] == []


@pytest.mark.parametrize("name", IMAGES)
def test_every_decoded_direct_branch_and_call_goes_where_objdump_says(
	decoded: Mapping[str, Decoded], name: str
) -> None:
	assert [
		(f"{instruction.address:#x}", listed.text, ours)
		for block in decoded[name].blocks
		for instruction in block.instructions
		for listed in (decoded[name].listing.get(Address(instruction.address)),)
		if listed is not None and listed.target is not None
		for ours in (_direct_target(instruction, decoded[name].program.machine),)
		if ours is None or ours & ~1 != listed.target & ~1
	] == []


@pytest.mark.parametrize("name", IMAGES)
def test_every_register_branch_objdump_lists_is_a_reported_site(
	decoded: Mapping[str, Decoded], name: str
) -> None:
	assert {
		address
		for address, listed in decoded[name].listing.items()
		for mnemonic, _, operands in (listed.text.partition(" "),)
		if _REGISTER_BRANCH.fullmatch(mnemonic) and _REGISTER.fullmatch(operands)
	} == decoded[name].sites


@pytest.mark.parametrize("name", IMAGES)
def test_every_other_write_to_pc_is_a_return_or_a_jump_table_the_decoder_follows(
	decoded: Mapping[str, Decoded], name: str
) -> None:
	successors = {
		block.instructions[-1].address: len(block.successors) for block in decoded[name].blocks
	}
	assert [
		(f"{address:#x}", listed.text)
		for address, listed in decoded[name].listing.items()
		if _PC_WRITE.match(listed.text)
		and not _RETURN.match(listed.text)
		and not (_TABLE.match(listed.text) and successors.get(address, 0) > 1)
	] == []
