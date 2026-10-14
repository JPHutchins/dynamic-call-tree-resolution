# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Control-flow graphs of function code."""

from dynamic_call_tree_resolution.vsa.cfg.blocks import Block as Block
from dynamic_call_tree_resolution.vsa.cfg.blocks import CallsNoReturn as CallsNoReturn
from dynamic_call_tree_resolution.vsa.cfg.blocks import function_graph as function_graph
from dynamic_call_tree_resolution.vsa.cfg.blocks import reachable_blocks as reachable_blocks
from dynamic_call_tree_resolution.vsa.cfg.graphs import control_flow_graphs as control_flow_graphs
from dynamic_call_tree_resolution.vsa.cfg.transfers import DirectCall as DirectCall
from dynamic_call_tree_resolution.vsa.cfg.transfers import DirectTransfer as DirectTransfer
from dynamic_call_tree_resolution.vsa.cfg.transfers import MemorySite as MemorySite
from dynamic_call_tree_resolution.vsa.cfg.transfers import RegisterSite as RegisterSite
from dynamic_call_tree_resolution.vsa.cfg.transfers import TailJump as TailJump
from dynamic_call_tree_resolution.vsa.cfg.transfers import branch_target as branch_target
from dynamic_call_tree_resolution.vsa.cfg.transfers import call_target as call_target
from dynamic_call_tree_resolution.vsa.cfg.transfers import direct_transfer as direct_transfer
from dynamic_call_tree_resolution.vsa.cfg.transfers import indirect_operand as indirect_operand
from dynamic_call_tree_resolution.vsa.cfg.transfers import is_returning_trap as is_returning_trap
