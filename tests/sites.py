# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""What the value-set analysis tracked into a call site, for assertions over many sites."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from dynamic_call_tree_resolution.model import Unreached
from dynamic_call_tree_resolution.vsa.lattice import Known, Top

if TYPE_CHECKING:
	from dynamic_call_tree_resolution import Address, CallSite


def tracked_values(site: CallSite) -> frozenset[Address]:
	match site.target:
		case Known(values=values):
			return values
		case Top() | Unreached():
			return frozenset()
		case _ as unreachable:
			assert_never(unreachable)
