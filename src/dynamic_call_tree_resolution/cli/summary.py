# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The ``summary`` and ``referrers`` commands, and the entry point."""

from pathlib import Path  # noqa: TC003  # cyclopts evaluates Annotated[Path, ...] at runtime
from typing import Annotated, assert_never

import msgspec
from cyclopts import Parameter
from elftools.common.exceptions import ELFError

from dynamic_call_tree_resolution.cli.app import (
	ASSUME_FRAME,
	ASSUME_NO_RECURSION,
	ASSUME_UNWRITTEN,
	NARROW_BY_FIELD,
	NARROW_BY_SIGNATURE,
	RTOS,
	app,
)
from dynamic_call_tree_resolution.cli.expansion import (
	elf_expansion,
)
from dynamic_call_tree_resolution.cli.stack import (
	in_image,
	own_stack_graphs,
	with_assumed_frames,
	with_interrupt_stack,
)
from dynamic_call_tree_resolution.linker import (
	Linkage,
	Membership,
)
from dynamic_call_tree_resolution.loader import (
	load,
)
from dynamic_call_tree_resolution.report import (
	AnalysisSummary,
	ReferrerReport,
	referrers_report,
	stack_bound_report,
)
from dynamic_call_tree_resolution.rtos import RtosChoice
from dynamic_call_tree_resolution.stack_analysis import (
	Bounded,
	StackReport,
	stack_graph,
	thread_reports,
)


@app.command  # type: ignore[misc]
def summary(
	build_directory: Path,
	elf: Path,
	*,
	narrow_by_signature: Annotated[bool, NARROW_BY_SIGNATURE] = False,
	narrow_by_field: Annotated[bool, NARROW_BY_FIELD] = False,
	assume_no_recursion: Annotated[tuple[str, ...], ASSUME_NO_RECURSION] = (),
	assume_unwritten: Annotated[tuple[str, ...], ASSUME_UNWRITTEN] = (),
	assume_frame: Annotated[tuple[str, ...], ASSUME_FRAME] = (),
	rtos: Annotated[RtosChoice, RTOS] = RtosChoice.AUTO,
) -> None:
	"""Print a JSON summary combining resolution rates and worst-case stack depth."""
	expansion = with_assumed_frames(
		elf_expansion(
			build_directory,
			elf,
			narrow_by_signature=narrow_by_signature,
			narrow_by_field=narrow_by_field,
			rtos=rtos,
			assume_unwritten=assume_unwritten,
		),
		assume_frame,
	)
	reports = with_interrupt_stack(
		expansion,
		stack_graph(
			expansion.expanded,
			expansion.frames,
			entry_edges=expansion.in_image_edges,
			assumed_no_recursion=frozenset(assume_no_recursion),
			assumed_unwritten=expansion.assumed_unwritten.callers,
			hardware_roots=expansion.hardware_roots,
		),
	)
	kept = in_image(
		thread_reports(
			reports,
			own_stack_graphs(expansion, frozenset(assume_no_recursion)),
			expansion.rtos,
		),
		expansion,
	)
	worst = next(iter(kept), StackReport(entry="", bound=Bounded(bytes=0)))
	report = AnalysisSummary(
		resolved_slots=expansion.counts.resolved_slots,
		total_slots=expansion.counts.total_slots,
		unresolved_slots=len(expansion.unresolved),
		resolved_targets=expansion.counts.resolved_targets,
		indirect_call_sites=expansion.indirect_sites,
		total_functions=len(expansion.program.functions),
		entry_points=len(kept),
		discarded_entry_points=len(reports)
		- len(kept)
		+ len(expansion.dropped.get(Linkage.DISCARDED, frozenset())),
		worst_case_entry=worst.entry,
		worst_case=stack_bound_report(worst.bound),
		rtos=expansion.rtos.name,
		membership=expansion.membership,
		never_linked_entry_points=(
			len(expansion.dropped[Linkage.NEVER_LINKED])
			if expansion.membership is Membership.LINKER
			else None
		),
		phantom_libcalls=(
			tuple(sorted(expansion.phantom_libcalls))
			if expansion.membership is Membership.LINKER
			else None
		),
	)
	print(msgspec.json.format(msgspec.json.encode(report).decode()))


@app.command  # type: ignore[misc]
def referrers(
	elf: Annotated[Path, Parameter(help="ELF image to analyze")],
	*,
	json: Annotated[bool, Parameter(name=("--json", "-j"), help="emit JSON")] = False,
) -> None:
	"""Print every address-taken function and each place that holds or computes its address."""
	report = referrers_report(load(elf))
	if json:
		print(msgspec.json.format(msgspec.json.encode(report).decode()))
		return
	for function in report:
		print(f"{function.name}: {', '.join(map(_render_referrer, function.referrers))}")


def _render_referrer(referrer: ReferrerReport) -> str:
	match referrer.holder:
		case None:
			return f"{referrer.slot:#x}"
		case str() as holder:
			return f"{holder}@{referrer.slot:#x}"
		case _ as unreachable:
			assert_never(unreachable)


def main() -> None:
	"""Entry point installed as the ``dctr`` executable.

	Raises:
		SystemExit: for usage errors and unparseable inputs.
	"""
	try:
		app()
	except (OSError, ELFError, ValueError, msgspec.ValidationError) as error:
		raise SystemExit(f"dctr: {error}") from None
