# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""What the loader records for the escape scan: code, load addresses, and parameterless prototypes."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest
from elftools.elf.elffile import ELFFile

from dynamic_call_tree_resolution import Address, load
from dynamic_call_tree_resolution.model import aligned
from tests.toolchains import FIXTURES

if TYPE_CHECKING:
	from pathlib import Path


def test_load_marks_code_sections_executable(fixture_elfs: dict[str, Path]) -> None:
	with fixture_elfs["nopie"].open("rb") as stream:
		starts = {
			section.name: Address(section.header.sh_addr)
			for section in ELFFile(stream).iter_sections()
		}
	sections = load(fixture_elfs["nopie"]).sections
	assert (sections[starts[".text"]].executable, sections[starts[".data"]].executable) == (
		True,
		False,
	)


@pytest.mark.parametrize(
	("source", "standard", "parameterless"),
	[
		pytest.param("prototypes.c", "gnu2x", ["empty", "none"], id="c23"),
		pytest.param("old_style.c", "gnu17", ["none"], id="c17"),
	],
)
def test_load_reads_which_functions_a_prototype_gives_no_parameters(
	tmp_path: Path, source: str, standard: str, parameterless: list[str]
) -> None:
	elf = tmp_path / "image.elf"
	subprocess.run(
		[
			"arm-none-eabi-gcc",
			f"-std={standard}",
			"-mcpu=cortex-m3",
			"-mthumb",
			"-nostdlib",
			"-g",
			"-e",
			"none",
			str(FIXTURES / source),
			"-o",
			str(elf),
		],
		check=True,
		capture_output=True,
	)
	program = load(elf)
	assert (
		sorted(
			function.name
			for function in program.functions.values()
			if aligned(function.address) in program.parameterless
		)
		== parameterless
	)


@pytest.mark.image
def test_load_reads_where_hello_holds_its_initialized_data_and_what_it_zeroes(
	zephyr_fixtures: Path,
) -> None:
	program = load(zephyr_fixtures / "hello" / "zephyr" / "zephyr.elf")
	assert (
		program.sections[Address(0x2000_0000)].load_address,
		program.sections[Address(0x2000_0040)].load_address,
		program.sections[Address(0)].load_address,
		program.nobits,
	) == (0x280C, 0x284C, None, ((0x2000_0048, 0x2000_0356), (0x2000_0358, 0x2000_1058)))
