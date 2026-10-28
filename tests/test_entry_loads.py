# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Which loads read through an argument before the function could have changed what it points to."""

import pytest
from capstone import arm_const

from dynamic_call_tree_resolution.model import Address, Machine
from dynamic_call_tree_resolution.vsa.cfg import control_flow_graphs
from dynamic_call_tree_resolution.vsa.entry_loads import EntryLoad, entry_loads
from tests.programs import build_program


def _loads(code: str, machine: Machine = Machine.EM_ARM) -> tuple[EntryLoad, ...]:
	body = bytes.fromhex(code)
	program = build_program(
		machine, (("callee", 0x1000, len(body)),), sections={0x1000: body}, pointer_size=4
	)
	return entry_loads(program, control_flow_graphs(program).values()).get(Address(0x1000), ())


def test_a_load_after_the_prologue_push_reads_through_its_argument() -> None:
	assert _loads("10b5" "4368" "1847") == (EntryLoad(Address(0x1002), arm_const.ARM_REG_R0, 4),)


@pytest.mark.parametrize(
	"code",
	[
		pytest.param("0160" "4368" "1847", id="after a store"),
		pytest.param("9047" "4368" "1847", id="after a call"),
		pytest.param("0130" "4368" "1847", id="after the argument changes"),
		pytest.param("08bf" "4368" "1847", id="predicated"),
		pytest.param("50f8043f" "1847", id="with writeback"),
		pytest.param("4368" "002b" "fcd1" "7047", id="in an entry block a branch returns to"),
	],
)
def test_a_load_is_no_entry_load_once_the_word_it_reads_could_have_changed(code: str) -> None:
	assert _loads(code) == ()


def test_an_x86_function_has_no_entry_loads() -> None:
	assert _loads("8b4704" "c3", Machine.EM_X86_64) == ()
