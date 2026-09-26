# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Static resolution of indirect calls in embedded firmware ELF images."""

from dynamic_call_tree_resolution.callgraph import CallEdge, load_callgraph, parse_callgraph
from dynamic_call_tree_resolution.cli import app, main
from dynamic_call_tree_resolution.loader import load
from dynamic_call_tree_resolution.model import (
	Address,
	DataObject,
	EmbeddedStructMember,
	Function,
	FunctionPointerMember,
	FunctionSignature,
	Member,
	Program,
	Provenance,
	Relocation,
	SlotAssignment,
	StructPointerMember,
	StructureLayout,
	render_path,
)
from dynamic_call_tree_resolution.points_to import assignments
from dynamic_call_tree_resolution.report import (
	AnalysisReport,
	Candidate,
	SlotAssignmentReport,
	build_report,
)
from dynamic_call_tree_resolution.stack_analysis import StackReport, worst_case_depths
from dynamic_call_tree_resolution.stack_usage import (
	StackUsage,
	load_stack_usages,
	parse_stack_usage,
)

__all__ = [
	"Address",
	"AnalysisReport",
	"CallEdge",
	"Candidate",
	"DataObject",
	"EmbeddedStructMember",
	"Function",
	"FunctionPointerMember",
	"FunctionSignature",
	"Member",
	"Program",
	"Provenance",
	"Relocation",
	"SlotAssignment",
	"SlotAssignmentReport",
	"StackReport",
	"StackUsage",
	"StructPointerMember",
	"StructureLayout",
	"app",
	"assignments",
	"build_report",
	"load",
	"load_callgraph",
	"load_stack_usages",
	"main",
	"parse_callgraph",
	"parse_stack_usage",
	"render_path",
	"worst_case_depths",
]
