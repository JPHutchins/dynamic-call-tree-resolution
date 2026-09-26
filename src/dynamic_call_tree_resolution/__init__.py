# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Static resolution of indirect calls in embedded firmware ELF images."""

from dynamic_call_tree_resolution.cli import app, main
from dynamic_call_tree_resolution.loader import load
from dynamic_call_tree_resolution.model import (
	Address,
	DataObject,
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

__all__ = [
	"Address",
	"AnalysisReport",
	"Candidate",
	"DataObject",
	"Function",
	"FunctionPointerMember",
	"FunctionSignature",
	"Member",
	"Program",
	"Provenance",
	"Relocation",
	"SlotAssignment",
	"SlotAssignmentReport",
	"StructPointerMember",
	"StructureLayout",
	"app",
	"assignments",
	"build_report",
	"load",
	"main",
	"render_path",
]
