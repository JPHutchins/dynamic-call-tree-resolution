# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""GCC's constant DWARF call values, against the value-set analysis's call arguments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.dwarf_call_values import checked

if TYPE_CHECKING:
	from pathlib import Path

pytestmark = pytest.mark.image


@pytest.mark.parametrize("image", ["hello", "sensor-two-impl", "sensor-threads", "synchronization"])
def test_every_constant_dwarf_call_value_is_an_argument_the_analysis_allows(
	image: str, zephyr_fixtures: Path
) -> None:
	result = checked(zephyr_fixtures / image / "zephyr" / "zephyr.elf")
	assert (result.excluded, result.agreed > 0) == ((), True)
