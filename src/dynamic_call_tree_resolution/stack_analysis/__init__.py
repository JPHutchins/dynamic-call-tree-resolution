# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Worst-case stack depth over a resolved call graph."""

from dynamic_call_tree_resolution.stack_analysis.edges import ThreadEntry as ThreadEntry
from dynamic_call_tree_resolution.stack_analysis.edges import ThreadTargets as ThreadTargets
from dynamic_call_tree_resolution.stack_analysis.edges import (
	expand_indirect_calls as expand_indirect_calls,
)
from dynamic_call_tree_resolution.stack_analysis.edges import own_thread_edges as own_thread_edges
from dynamic_call_tree_resolution.stack_analysis.edges import thread_edges as thread_edges
from dynamic_call_tree_resolution.stack_analysis.edges import thread_frames as thread_frames
from dynamic_call_tree_resolution.stack_analysis.graph import stack_graph as stack_graph
from dynamic_call_tree_resolution.stack_analysis.paths import deepest_path as deepest_path
from dynamic_call_tree_resolution.stack_analysis.reports import entry_report as entry_report
from dynamic_call_tree_resolution.stack_analysis.reports import (
	interrupt_stack_report as interrupt_stack_report,
)
from dynamic_call_tree_resolution.stack_analysis.reports import stack_reports as stack_reports
from dynamic_call_tree_resolution.stack_analysis.reports import thread_reports as thread_reports
from dynamic_call_tree_resolution.stack_analysis.reports import (
	worst_case_depths as worst_case_depths,
)
from dynamic_call_tree_resolution.stack_analysis.values import INDIRECT_CALLEE as INDIRECT_CALLEE
from dynamic_call_tree_resolution.stack_analysis.values import NO_CALLEE as NO_CALLEE
from dynamic_call_tree_resolution.stack_analysis.values import AssumedFrame as AssumedFrame
from dynamic_call_tree_resolution.stack_analysis.values import Bounded as Bounded
from dynamic_call_tree_resolution.stack_analysis.values import Frame as Frame
from dynamic_call_tree_resolution.stack_analysis.values import FunctionReason as FunctionReason
from dynamic_call_tree_resolution.stack_analysis.values import Handled as Handled
from dynamic_call_tree_resolution.stack_analysis.values import LevelSource as LevelSource
from dynamic_call_tree_resolution.stack_analysis.values import MeasuredFrame as MeasuredFrame
from dynamic_call_tree_resolution.stack_analysis.values import Nested as Nested
from dynamic_call_tree_resolution.stack_analysis.values import Nesting as Nesting
from dynamic_call_tree_resolution.stack_analysis.values import OwnFrame as OwnFrame
from dynamic_call_tree_resolution.stack_analysis.values import PathStep as PathStep
from dynamic_call_tree_resolution.stack_analysis.values import PriorityLevels as PriorityLevels
from dynamic_call_tree_resolution.stack_analysis.values import Reason as Reason
from dynamic_call_tree_resolution.stack_analysis.values import StackGraph as StackGraph
from dynamic_call_tree_resolution.stack_analysis.values import StackReport as StackReport
from dynamic_call_tree_resolution.stack_analysis.values import Unbounded as Unbounded
from dynamic_call_tree_resolution.stack_analysis.values import frame_key as frame_key
