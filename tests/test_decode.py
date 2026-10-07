# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The analysis decodes the Arm Zephyr fixtures as GNU objdump does (#75, item 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from tests.decode_oracle import (
	Decoded,
	decoded,
	misaligned,
	misdirected,
	register_branches,
	unexplained_pc_writes,
	unreadable,
)

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path

pytestmark = pytest.mark.image

IMAGES: Final = ["hello", "sensor-threads", "sensor-two-impl", "synchronization"]


@pytest.fixture(scope="module")
def images(zephyr_fixtures: Path) -> Mapping[str, Decoded]:
	return {name: decoded(zephyr_fixtures / name / "zephyr" / "zephyr.elf") for name in IMAGES}


@pytest.mark.parametrize("name", IMAGES)
def test_every_decoded_instruction_starts_where_objdump_starts_one(
	images: Mapping[str, Decoded], name: str
) -> None:
	assert misaligned(images[name]) == []


@pytest.mark.parametrize("name", IMAGES)
def test_no_decoded_instruction_is_bytes_the_decoder_could_not_read(
	images: Mapping[str, Decoded], name: str
) -> None:
	assert unreadable(images[name]) == []


@pytest.mark.parametrize("name", IMAGES)
def test_every_decoded_direct_branch_and_call_goes_where_objdump_says(
	images: Mapping[str, Decoded], name: str
) -> None:
	assert misdirected(images[name]) == []


@pytest.mark.parametrize("name", IMAGES)
def test_every_register_branch_objdump_lists_is_a_reported_site(
	images: Mapping[str, Decoded], name: str
) -> None:
	assert register_branches(images[name]) == images[name].sites


@pytest.mark.parametrize("name", IMAGES)
def test_every_other_write_to_pc_is_a_return_or_a_jump_table_the_decoder_follows(
	images: Mapping[str, Decoded], name: str
) -> None:
	assert unexplained_pc_writes(images[name]) == []
