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
	UnresolvedSlot,
	render_path,
)
from dynamic_call_tree_resolution.pexplorer import (
	PexplorerCallee,
	PexplorerFunction,
	PexplorerReport,
	load_pexplorer,
)
from dynamic_call_tree_resolution.points_to import assignments, unresolved_slots
from dynamic_call_tree_resolution.report import (
	AnalysisReport,
	AnalysisSummary,
	CallSiteReport,
	Candidate,
	ComparisonReport,
	FunctionComparison,
	SignatureReport,
	SlotAssignmentReport,
	UnresolvedSlotReport,
	build_comparison,
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
	"ComparisonReport",
	"DataObject",
	"EmbeddedStructMember",
	"Function",
	"FunctionComparison",
	"FunctionPointerMember",
	"FunctionSignature",
	"Member",
	"PexplorerCallee",
	"PexplorerFunction",
	"PexplorerReport",
	"Program",
	"Provenance",
	"Relocation",
	"SignatureReport",
	"SlotAssignment",
	"SlotAssignmentReport",
	"StackReport",
	"StackUsage",
	"StructPointerMember",
	"StructureLayout",
	"UnresolvedSlot",
	"UnresolvedSlotReport",
	"app",
	"assignments",
	"build_comparison",
	"build_report",
	"call_site_candidates",
	"expand_indirect_calls",
	"extract_call_sites",
	"load",
	"load_callgraph",
	"load_pexplorer",
	"load_stack_usages",
	"main",
	"parse_callgraph",
	"parse_stack_usage",
	"per_caller_candidates",
	"render_path",
	"unresolved_slots",
	"worst_case_depths",
]
