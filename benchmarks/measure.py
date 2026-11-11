# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""One measurement of dctr's value-set analysis of one ELF image, in a process of its own.

``benchmarks.compare`` runs it with ``PYTHONPATH`` naming the source tree it measures. The
analysis runs under the RTOS model ``dctr stack`` detects, from the image and its build's
Kconfig; measuring the analysis leaves out loading the image and detecting the model.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import Counter
from pathlib import Path  # noqa: TC003  # cyclopts evaluates the annotations at runtime
from typing import TYPE_CHECKING, Annotated, Literal, assert_never

import msgspec
from cyclopts import App, Parameter

from benchmarks.measurements import CpuTime, Work
from dynamic_call_tree_resolution.cli.measures import build_kconfig
from dynamic_call_tree_resolution.loader import load
from dynamic_call_tree_resolution.rtos import RtosChoice, rtos_model
from dynamic_call_tree_resolution.vsa import analyze
from dynamic_call_tree_resolution.vsa.interpret import analyze_function, transfer

if TYPE_CHECKING:
	from types import CodeType

	from dynamic_call_tree_resolution.model import Program, RtosModel

app = App(name="measure")


@app.default  # type: ignore[misc]
def measure(
	kind: Annotated[
		Literal["cpu", "load", "work"],
		Parameter(
			help="cpu: the process CPU seconds of the analysis. load: the process CPU seconds of "
			"loading the image. work: the instructions the interpreter transferred and the "
			"functions it analyzed, counted on their code objects so that every caller and every "
			"thread is seen"
		),
	],
	elf: Annotated[Path, Parameter(help="ELF image to analyze")],
) -> None:
	"""Print one measurement of loading or analyzing an ELF image as a JSON line."""
	print(msgspec.json.encode(_measured(kind, elf)).decode())


def _measured(kind: Literal["cpu", "load", "work"], elf: Path) -> CpuTime | Work:
	match kind:
		case "cpu":
			program = load(elf)
			return _cpu_time(program, _model(program, elf))
		case "load":
			return _load_time(elf)
		case "work":
			program = load(elf)
			return _work(program, _model(program, elf))
		case _ as unreachable:
			assert_never(unreachable)


def _model(program: Program, elf: Path) -> RtosModel:
	return rtos_model(program, RtosChoice.AUTO, build_kconfig(elf))


def _load_time(elf: Path) -> CpuTime:
	started = time.process_time()
	load(elf)
	return CpuTime(time.process_time() - started)


def _cpu_time(program: Program, model: RtosModel) -> CpuTime:
	started = time.process_time()
	analyze(program, model)
	return CpuTime(time.process_time() - started)


def _work(program: Program, model: RtosModel) -> Work:
	lock = threading.Lock()
	starts: Counter[CodeType] = Counter()

	def started(code: CodeType, _instruction_offset: int) -> None:
		with lock:
			starts[code] += 1

	sys.monitoring.use_tool_id(sys.monitoring.PROFILER_ID, "dctr benchmarks")
	sys.monitoring.register_callback(
		sys.monitoring.PROFILER_ID, sys.monitoring.events.PY_START, started
	)
	sys.monitoring.set_local_events(
		sys.monitoring.PROFILER_ID, transfer.__code__, sys.monitoring.events.PY_START
	)
	sys.monitoring.set_local_events(
		sys.monitoring.PROFILER_ID, analyze_function.__code__, sys.monitoring.events.PY_START
	)
	analyze(program, model)
	return Work(starts[transfer.__code__], starts[analyze_function.__code__])


if __name__ == "__main__":
	app()
