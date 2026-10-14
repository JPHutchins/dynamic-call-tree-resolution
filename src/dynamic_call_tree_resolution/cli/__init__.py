# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Command line interface for analyzing ELF images."""

from dynamic_call_tree_resolution.cli.analyze import analyze as analyze
from dynamic_call_tree_resolution.cli.analyze import compare as compare
from dynamic_call_tree_resolution.cli.app import app as app
from dynamic_call_tree_resolution.cli.artifacts import AssumedUnwritten as AssumedUnwritten
from dynamic_call_tree_resolution.cli.artifacts import Expansion as Expansion
from dynamic_call_tree_resolution.cli.expansion import elf_expansion as elf_expansion
from dynamic_call_tree_resolution.cli.stack import stack as stack
from dynamic_call_tree_resolution.cli.summary import main as main
from dynamic_call_tree_resolution.cli.summary import referrers as referrers
from dynamic_call_tree_resolution.cli.summary import summary as summary
