# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Loading ELF and DWARF data into the immutable :class:`Program` model."""

from dynamic_call_tree_resolution.loader.functions import (
	defined_function_names as defined_function_names,
)
from dynamic_call_tree_resolution.loader.program import elf_machine as elf_machine
from dynamic_call_tree_resolution.loader.program import load as load
from dynamic_call_tree_resolution.loader.relocations import ArmRelocation as ArmRelocation
from dynamic_call_tree_resolution.loader.relocations import I386Relocation as I386Relocation
from dynamic_call_tree_resolution.loader.relocations import X64Relocation as X64Relocation
from dynamic_call_tree_resolution.loader.spans import line_spans as line_spans
