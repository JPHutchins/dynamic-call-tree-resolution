# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The JSON line ``benchmarks.measure`` prints and ``benchmarks.compare`` reads."""

from msgspec import Struct


class CpuTime(Struct, frozen=True, forbid_unknown_fields=True):
	"""The process CPU time of one measured stage, every thread summed."""

	cpu_seconds: float


class Work(Struct, frozen=True, forbid_unknown_fields=True):
	"""What one analysis did, counted so that machine load can't change it."""

	transfers: int
	"""Instructions the interpreter transferred."""
	function_analyses: int
