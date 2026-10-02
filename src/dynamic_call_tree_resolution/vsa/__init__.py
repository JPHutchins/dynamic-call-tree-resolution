# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Whole-function value-set analysis of indirect call sites."""

from dynamic_call_tree_resolution.vsa.analysis import Analysis as Analysis
from dynamic_call_tree_resolution.vsa.analysis import analyze as analyze
from dynamic_call_tree_resolution.vsa.fallback import address_taken as address_taken
from dynamic_call_tree_resolution.vsa.fallback import (
	linked_address_taken as linked_address_taken,
)
from dynamic_call_tree_resolution.vsa.fallback import referrers as referrers
from dynamic_call_tree_resolution.vsa.links import LinkedCall as LinkedCall
from dynamic_call_tree_resolution.vsa.links import linked_calls as linked_calls
from dynamic_call_tree_resolution.vsa.memory import runtime_value as runtime_value
