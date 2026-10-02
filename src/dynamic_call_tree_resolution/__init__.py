# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Static resolution of indirect calls in embedded firmware ELF images."""

from dynamic_call_tree_resolution.call_sites import (
	ProgramResolution as ProgramResolution,
)
from dynamic_call_tree_resolution.call_sites import (
	call_site_candidates as call_site_candidates,
)
from dynamic_call_tree_resolution.call_sites import (
	extract_call_sites as extract_call_sites,
)
from dynamic_call_tree_resolution.call_sites import (
	matching_targets as matching_targets,
)
from dynamic_call_tree_resolution.call_sites import (
	per_caller_candidates as per_caller_candidates,
)
from dynamic_call_tree_resolution.call_sites import (
	resolve as resolve,
)
from dynamic_call_tree_resolution.callgraph import (
	CallEdge as CallEdge,
)
from dynamic_call_tree_resolution.callgraph import (
	EdgeKind as EdgeKind,
)
from dynamic_call_tree_resolution.callgraph import (
	load_callgraph as load_callgraph,
)
from dynamic_call_tree_resolution.callgraph import (
	parse_callgraph as parse_callgraph,
)
from dynamic_call_tree_resolution.cli import app as app
from dynamic_call_tree_resolution.cli import main as main
from dynamic_call_tree_resolution.loader import load as load
from dynamic_call_tree_resolution.model import (
	Address as Address,
)
from dynamic_call_tree_resolution.model import (
	ArrayMember as ArrayMember,
)
from dynamic_call_tree_resolution.model import (
	CallSite as CallSite,
)
from dynamic_call_tree_resolution.model import (
	DataObject as DataObject,
)
from dynamic_call_tree_resolution.model import (
	EmbeddedStructMember as EmbeddedStructMember,
)
from dynamic_call_tree_resolution.model import (
	Function as Function,
)
from dynamic_call_tree_resolution.model import (
	FunctionPointerMember as FunctionPointerMember,
)
from dynamic_call_tree_resolution.model import (
	FunctionSignature as FunctionSignature,
)
from dynamic_call_tree_resolution.model import (
	LinkReference as LinkReference,
)
from dynamic_call_tree_resolution.model import (
	Machine as Machine,
)
from dynamic_call_tree_resolution.model import (
	Member as Member,
)
from dynamic_call_tree_resolution.model import (
	NotEnumerated as NotEnumerated,
)
from dynamic_call_tree_resolution.model import (
	Program as Program,
)
from dynamic_call_tree_resolution.model import (
	Provenance as Provenance,
)
from dynamic_call_tree_resolution.model import (
	ReferenceKind as ReferenceKind,
)
from dynamic_call_tree_resolution.model import (
	Relocation as Relocation,
)
from dynamic_call_tree_resolution.model import (
	Residue as Residue,
)
from dynamic_call_tree_resolution.model import (
	Section as Section,
)
from dynamic_call_tree_resolution.model import (
	SkippedMember as SkippedMember,
)
from dynamic_call_tree_resolution.model import (
	SkipReason as SkipReason,
)
from dynamic_call_tree_resolution.model import (
	SlotAssignment as SlotAssignment,
)
from dynamic_call_tree_resolution.model import (
	StructPointerMember as StructPointerMember,
)
from dynamic_call_tree_resolution.model import (
	StructureLayout as StructureLayout,
)
from dynamic_call_tree_resolution.model import (
	UnresolvedSlot as UnresolvedSlot,
)
from dynamic_call_tree_resolution.model import (
	render_path as render_path,
)
from dynamic_call_tree_resolution.pexplorer import (
	PexplorerCallee as PexplorerCallee,
)
from dynamic_call_tree_resolution.pexplorer import (
	PexplorerFunction as PexplorerFunction,
)
from dynamic_call_tree_resolution.pexplorer import (
	PexplorerReport as PexplorerReport,
)
from dynamic_call_tree_resolution.pexplorer import (
	load_pexplorer as load_pexplorer,
)
from dynamic_call_tree_resolution.points_to import (
	assignments as assignments,
)
from dynamic_call_tree_resolution.points_to import (
	not_enumerated as not_enumerated,
)
from dynamic_call_tree_resolution.points_to import (
	unresolved_slots as unresolved_slots,
)
from dynamic_call_tree_resolution.report import (
	AnalysisReport as AnalysisReport,
)
from dynamic_call_tree_resolution.report import (
	AnalysisSummary as AnalysisSummary,
)
from dynamic_call_tree_resolution.report import (
	BoundedStack as BoundedStack,
)
from dynamic_call_tree_resolution.report import (
	CallSiteReport as CallSiteReport,
)
from dynamic_call_tree_resolution.report import (
	Candidate as Candidate,
)
from dynamic_call_tree_resolution.report import (
	ComparisonReport as ComparisonReport,
)
from dynamic_call_tree_resolution.report import (
	FunctionComparison as FunctionComparison,
)
from dynamic_call_tree_resolution.report import (
	NotEnumeratedReport as NotEnumeratedReport,
)
from dynamic_call_tree_resolution.report import (
	PathStepReport as PathStepReport,
)
from dynamic_call_tree_resolution.report import (
	SignatureReport as SignatureReport,
)
from dynamic_call_tree_resolution.report import (
	SlotAssignmentReport as SlotAssignmentReport,
)
from dynamic_call_tree_resolution.report import (
	StackEntryReport as StackEntryReport,
)
from dynamic_call_tree_resolution.report import (
	StackPathReport as StackPathReport,
)
from dynamic_call_tree_resolution.report import (
	UnboundedStack as UnboundedStack,
)
from dynamic_call_tree_resolution.report import (
	UnresolvedSlotReport as UnresolvedSlotReport,
)
from dynamic_call_tree_resolution.report import (
	build_comparison as build_comparison,
)
from dynamic_call_tree_resolution.report import (
	build_report as build_report,
)
from dynamic_call_tree_resolution.stack_analysis import (
	Bounded as Bounded,
)
from dynamic_call_tree_resolution.stack_analysis import (
	PathStep as PathStep,
)
from dynamic_call_tree_resolution.stack_analysis import (
	Reason as Reason,
)
from dynamic_call_tree_resolution.stack_analysis import (
	StackGraph as StackGraph,
)
from dynamic_call_tree_resolution.stack_analysis import (
	StackReport as StackReport,
)
from dynamic_call_tree_resolution.stack_analysis import (
	Unbounded as Unbounded,
)
from dynamic_call_tree_resolution.stack_analysis import (
	deepest_path as deepest_path,
)
from dynamic_call_tree_resolution.stack_analysis import (
	expand_indirect_calls as expand_indirect_calls,
)
from dynamic_call_tree_resolution.stack_analysis import (
	stack_graph as stack_graph,
)
from dynamic_call_tree_resolution.stack_analysis import (
	stack_reports as stack_reports,
)
from dynamic_call_tree_resolution.stack_analysis import (
	worst_case_depths as worst_case_depths,
)
from dynamic_call_tree_resolution.stack_usage import (
	StackUsage as StackUsage,
)
from dynamic_call_tree_resolution.stack_usage import (
	load_stack_usages as load_stack_usages,
)
from dynamic_call_tree_resolution.stack_usage import (
	parse_stack_usage as parse_stack_usage,
)
