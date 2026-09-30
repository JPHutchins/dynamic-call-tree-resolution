# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The Zephyr builds of `nix build .#fixtures`, which `nix develop` exports as DCTR_FIXTURES."""

from pathlib import Path
from typing import TYPE_CHECKING, assert_never

import pytest

if TYPE_CHECKING:
	from collections.abc import Mapping


def zephyr_fixtures_root(environment: Mapping[str, str]) -> Path:
	match environment.get("DCTR_FIXTURES"):
		case None | "":
			pytest.fail(
				"DCTR_FIXTURES is unset: the tests marked image read `nix build .#fixtures`, "
				"which `nix develop` exports as DCTR_FIXTURES"
			)
		case str() as root:
			return Path(root)
		case _ as unreachable:
			assert_never(unreachable)
