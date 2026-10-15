# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Each branch edge carries only the register values that take it (#185)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dynamic_call_tree_resolution import Machine, extract_call_sites
from tests.programs import build_program
from tests.sites import tracked_values

if TYPE_CHECKING:
	from dynamic_call_tree_resolution import Program


def _table_program(code: str) -> Program:
	"""A walk at 0x1000 over three handlers at 0x2000, then a fourth just past the table's end."""
	return build_program(
		Machine.EM_ARM,
		(
			("walk", 0x1000, len(bytes.fromhex(code))),
			*((f"handler_{index}", 0x3000 + 0x10 * index, 2) for index in range(4)),
		),
		objects=(
			(
				"table",
				0x2000,
				b"".join((0x3001 + 0x10 * index).to_bytes(4, "little") for index in range(3)),
			),
			("after_table", 0x200C, (0x3031).to_bytes(4, "little")),
		),
		sections={
			0x1000: bytes.fromhex(code),
			0x3000: bytes.fromhex("7047" + "00" * 14) * 4,
		},
		pointer_size=4,
	)


@pytest.mark.parametrize(
	"code",
	[
		pytest.param(
			"42f2000442f20c05ac4203d2236898470434f9e77047",
			id="a pointer compared at the loop's head",
		),
		pytest.param(
			"30b542f2000442f20c05ac4228bf30bd236898470434f8e7",
			id="a predicated return leaves the loop",
		),
		pytest.param(
			"42f200060024032c04d256f8243098470134f8e77047",
			id="an index compared against an immediate",
		),
	],
)
def test_a_walk_calls_only_the_entries_before_its_bound(code: str) -> None:
	(site,) = extract_call_sites(_table_program(code))
	assert tracked_values(site) == {0x3001, 0x3011, 0x3021}


def test_a_cbz_keeps_null_off_the_call_it_guards() -> None:
	code = bytes.fromhex("0023002818bf43f2010303b198477047")
	(site,) = extract_call_sites(
		build_program(
			Machine.EM_ARM,
			(("guarded", 0x1000, len(code)), ("handler", 0x3000, 2)),
			sections={0x1000: code, 0x3000: bytes.fromhex("7047")},
			pointer_size=4,
		)
	)
	assert tracked_values(site) == {0x3001}
