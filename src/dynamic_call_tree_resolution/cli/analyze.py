# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The ``analyze`` and ``compare`` commands."""

import sys
from pathlib import Path  # noqa: TC003  # cyclopts evaluates Annotated[Path, ...] at runtime
from typing import TYPE_CHECKING, Annotated

import msgspec
from cyclopts import Parameter

from dynamic_call_tree_resolution.call_sites import (
	resolve,
)
from dynamic_call_tree_resolution.cli.app import (
	NARROW_BY_FIELD,
	NARROW_BY_SIGNATURE,
	RTOS,
	app,
)
from dynamic_call_tree_resolution.cli.expansion import (
	field_narrowed_spans,
)
from dynamic_call_tree_resolution.cli.render import (
	keep,
	residue_label,
	rtos_lines,
	signature_text,
)
from dynamic_call_tree_resolution.loader import (
	load,
)
from dynamic_call_tree_resolution.pexplorer import PexplorerReport, load_pexplorer
from dynamic_call_tree_resolution.report import (
	CallSiteReport,
	ComparisonReport,
	build_comparison,
	build_report,
	rtos_report,
)
from dynamic_call_tree_resolution.rtos import RtosChoice, rtos_model

if TYPE_CHECKING:
	from collections.abc import Iterator


@app.command  # type: ignore[misc]
def analyze(
	elf: Annotated[Path, Parameter(help="ELF image to analyze")],
	*,
	json: Annotated[bool, Parameter(name=("--json", "-j"), help="emit JSON")] = False,
	narrow_by_signature: Annotated[bool, NARROW_BY_SIGNATURE] = False,
	narrow_by_field: Annotated[bool, NARROW_BY_FIELD] = False,
	rtos: Annotated[RtosChoice, RTOS] = RtosChoice.AUTO,
) -> None:
	"""Print resolved function-pointer assignments and call sites of an ELF image."""
	program = load(elf)
	model = rtos_model(program, rtos)
	resolution = resolve(program, model)
	report = build_report(
		program,
		resolution.assignments,
		resolution.sites,
		narrow_by_signature=narrow_by_signature,
		narrowed_by_field=field_narrowed_spans(elf, program) if narrow_by_field else (),
		rtos=rtos_report(program, model, resolution.seeded),
		fallback=resolution.fallback,
	)
	if json:
		print(msgspec.json.format(msgspec.json.encode(report).decode()))
		return
	for line in rtos_lines(report.rtos):
		print(line)
	for assignment in report.assignments:
		label = assignment.member_path or hex(assignment.slot_address)
		print(f"{label}: {', '.join(candidate.name for candidate in assignment.candidates)}")
	for null_slot in report.null_slots:
		print(f"{null_slot.member_path}: <null>")
	for slot in report.unresolved_slots:
		print(f"{slot.member_path}: {residue_label(slot.residue)}")
	for item in report.not_enumerated:
		print(f"{item.member_path}: <not enumerated: {item.reason}>")
	for site in report.call_sites:
		label = f"{site.caller}@{site.site_address:#x}"
		targets = ", ".join(candidate.name for candidate in site.candidates)
		narrowing = (
			f" (narrowed by field {site.field}, unsound under casts)"
			if site.field is not None
			else f" (narrowed by signature {signature_text(site.signature)}, unsound under casts)"
			if site.signature is not None
			else " (dispatch, from the RTOS model)"
			if site.dispatch
			else ""
		)
		print(f"{label}: {targets or _no_candidates(site)}{narrowing}")


def _no_candidates(site: CallSiteReport) -> str:
	return (
		f"<external: {site.external}>"
		if site.external is not None
		else "<null>"
		if site.null
		else "<dead>"
		if site.dead
		else "<unresolved>"
	)


@app.command  # type: ignore[misc]
def compare(
	elfs: Annotated[
		list[Path], Parameter(help="ELF images or directories of ELF images to compare")
	],
	*,
	json: Annotated[bool, Parameter(name=("--json", "-j"), help="emit JSON")] = False,
	pexplorer: Annotated[
		Path | None,
		Parameter(
			help="pexplorer JSON report to join per function: every function with dynamic "
			"calls on either side gets a row comparing pexplorer's dynamic-call count with "
			"dctr's per-site candidate sets"
		),
	] = None,
	narrow_by_signature: Annotated[bool, NARROW_BY_SIGNATURE] = False,
	narrow_by_field: Annotated[bool, NARROW_BY_FIELD] = False,
	rtos: Annotated[RtosChoice, RTOS] = RtosChoice.AUTO,
) -> None:
	"""Print a resolution rollup per ELF for cross-tool comparison."""
	paths = [path for path in tuple(_elf_paths(elfs)) if keep(path)]
	pexplorer_report = load_pexplorer(pexplorer) if pexplorer is not None else None
	comparisons = [
		_comparison(
			elf,
			pexplorer_report,
			narrow_by_signature=narrow_by_signature,
			narrow_by_field=narrow_by_field,
			rtos=rtos,
		)
		for elf in paths
	]
	if json:
		print(msgspec.json.format(msgspec.json.encode(comparisons).decode()))
		return
	print(f"{'elf':<46} {'machine':<10} {'functions':>9} {'slots r/u/t':>11} {'sites r/e/t':>12}")
	for comparison in comparisons:
		print(
			f"{comparison.elf:<46} {comparison.machine:<10} {comparison.functions:>9} "
			f"{f'{comparison.resolved_slots}/{comparison.unresolved_slots}/{comparison.total_slots}':>11} "
			f"{f'{comparison.resolved_call_sites}/{comparison.exact_call_sites}/{comparison.call_sites}':>12}"
		)
	if pexplorer_report is None:
		return
	print()
	print(f"{'function':<48} {'pexplorer':>9} {'dctr':>5} {'resolved':>8} {'exact':>5}")
	for comparison in comparisons:
		for row in comparison.function_comparisons:
			print(
				f"{row.caller:<48} {row.pexplorer_dynamic_sites:>9} "
				f"{row.dctr_call_sites:>5} {row.dctr_resolved_sites:>8} {row.dctr_exact_sites:>5}"
			)


def _elf_paths(elfs: list[Path]) -> Iterator[Path]:
	for elf in elfs:
		if elf.is_dir():
			found = sorted(elf.glob("*.elf"))
			if not found:
				print(f"warning: {elf}: no .elf files", file=sys.stderr)
			yield from found
		else:
			yield elf


def _comparison(
	elf: Path,
	pexplorer: PexplorerReport | None,
	*,
	narrow_by_signature: bool,
	narrow_by_field: bool,
	rtos: RtosChoice,
) -> ComparisonReport:
	program = load(elf)
	return build_comparison(
		elf.name,
		program,
		pexplorer,
		narrow_by_signature=narrow_by_signature,
		narrowed_by_field=field_narrowed_spans(elf, program) if narrow_by_field else (),
		rtos=rtos_model(program, rtos),
	)
