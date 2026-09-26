# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Static resolution of indirect calls in embedded firmware ELF images."""

from dynamic_call_tree_resolution.call_sites import (
	call_site_candidates,
	extract_call_sites,
	per_caller_candidates,
)
from dynamic_call_tree_resolution.callgraph import CallEdge, load_callgraph, parse_callgraph
from dynamic_call_tree_resolution.cli import app, main
from dynamic_call_tree_resolution.loader import load
from dynamic_call_tree_resolution.model import (
	Address,
	CallSite,
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
	AnalysisSummary,
	CallSiteReport,
	Candidate,
	SlotAssignmentReport,
	build_report,
)
from dynamic_call_tree_resolution.stack_analysis import (
	StackReport,
	expand_indirect_calls,
	worst_case_depths,
)
from dynamic_call_tree_resolution.stack_usage import (
	StackUsage,
	load_stack_usages,
	parse_stack_usage,
)

__all__ = [
	"Address",
	"AnalysisReport",
	"AnalysisSummary",
	"CallEdge",
	"CallSite",
	"CallSiteReport",
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
	"call_site_candidates",
	"expand_indirect_calls",
	"extract_call_sites",
	"load",
	"load_callgraph",
	"load_stack_usages",
	"main",
	"parse_callgraph",
	"parse_stack_usage",
	"per_caller_candidates",
	"render_path",
	"worst_case_depths",
]
