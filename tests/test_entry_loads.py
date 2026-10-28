# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Which loads read through an argument before the function could have changed what it points to."""

import pytest
from capstone import arm_const

from dynamic_call_tree_resolution.model import Address, Machine
from dynamic_call_tree_resolution.vsa.cfg import control_flow_graphs
from dynamic_call_tree_resolution.vsa.entry_loads import EntryLoad, entry_loads
from tests.programs import build_program


def _loads(
	code: str, machine: Machine = Machine.EM_ARM, *, a32: bool = False
) -> tuple[EntryLoad, ...]:
	body = bytes.fromhex(code)
	program = build_program(
		machine,
		(("callee", 0x1000, len(body)),),
		sections={0x1000: body},
		pointer_size=4,
		arm_code=((0x1000, 0x1000 + len(body)),) if a32 else (),
	)
	return entry_loads(program, control_flow_graphs(program).values()).get(Address(0x1000), ())


def test_a_load_after_the_prologue_push_reads_through_its_argument() -> None:
	assert _loads("10b543681847") == (EntryLoad(Address(0x1002), arm_const.ARM_REG_R0, 4),)


@pytest.mark.parametrize(
	"code",
	[
		pytest.param("016043681847", id="after a store"),
		pytest.param("904743681847", id="after a call"),
		pytest.param("013043681847", id="after the argument changes"),
		pytest.param("08bf43681847", id="predicated"),
		pytest.param("50f8043f1847", id="with writeback"),
		pytest.param("4368002bfcd17047", id="in an entry block a branch returns to"),
	],
)
def test_a_load_is_no_entry_load_once_the_word_it_reads_could_have_changed(code: str) -> None:
	assert _loads(code) == ()


def test_an_a32_load_after_the_prologue_push_reads_through_its_argument() -> None:
	assert _loads("10402de9043090e513ff2fe1", a32=True) == (
		EntryLoad(Address(0x1004), arm_const.ARM_REG_R0, 4),
	)


@pytest.mark.parametrize(
	"code",
	[
		pytest.param("1f056df9043090e513ff2fe1", id="after srsdb onto another mode's stack"),
		pytest.param("921000e1043090e513ff2fe1", id="after swp"),
	],
)
def test_an_a32_load_is_no_entry_load_after_a_store_the_store_tables_leave_out(code: str) -> None:
	assert _loads(code, a32=True) == ()


def test_an_x86_function_has_no_entry_loads() -> None:
	assert _loads("8b4704c3", Machine.EM_X86_64) == ()
