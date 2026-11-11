# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Reading words, pointers and instruction sets from an image."""

from __future__ import annotations

from bisect import bisect_right
from itertools import chain, pairwise
from typing import TYPE_CHECKING, assert_never

from salix import Struct

from dynamic_call_tree_resolution.model import (
	Address,
	InstructionSet,
	Machine,
	Program,
	Section,
)
from dynamic_call_tree_resolution.points_to.objects import object_covering

if TYPE_CHECKING:
	from collections.abc import Iterator

	from dynamic_call_tree_resolution.span import Span


def pointer_at(program: Program, address: Address) -> Address | None:
	covering = object_covering(program, address)
	bound = covering.address + max(covering.size, 1) if covering is not None else None
	return read_pointer(program, address, bound)


def read_pointer(program: Program, address: Address, bound: int | None) -> Address | None:
	if bound is not None and address + program.pointer_size > bound:
		return None
	data = memory_at(program, address, program.pointer_size)
	if len(data) != program.pointer_size:
		return None
	return Address(int.from_bytes(data, program.byte_order))


def in_writable_memory(program: Program, address: Address) -> bool:
	covering = _section_covering(program, address)
	return covering is None or covering.section.writable


def memory_at(program: Program, address: Address, size: int) -> bytes:
	covering = _section_covering(program, address)
	if covering is None:
		return b""
	return covering.section.data[address - covering.start : address - covering.start + size]


class _Covering(Struct):
	"""A section, with the address it starts at."""

	start: Address
	section: Section


def _section_covering(program: Program, address: Address) -> _Covering | None:
	return next(
		(
			_Covering(start=start, section=section)
			for start, section in program.sections.items()
			if start <= address < start + len(section.data)
		),
		None,
	)


class InstructionRun(Struct):
	"""Code of one instruction set, with the address it starts at."""

	start: Address
	code: bytes


def instruction_runs(program: Program, start: Address, size: int) -> tuple[InstructionRun, ...]:
	end = start + size
	data_edges = chain.from_iterable(
		(max(span.start, start), min(span.end, end))
		for span in sorted(program.data_in_code)
		if span.start < end and start < span.end
	)
	edges = tuple(chain((start,), data_edges, (end,)))
	return tuple(
		chain.from_iterable(
			_code_runs(program, code_start, code_end)
			for code_start, code_end in zip(edges[::2], edges[1::2], strict=True)
		)
	)


def _code_runs(program: Program, code_start: int, code_end: int) -> Iterator[InstructionRun]:
	arm_edges = chain.from_iterable((span.start, span.end) for span in program.arm_code)
	inner_edges = sorted(edge for edge in arm_edges if code_start < edge < code_end)
	return (
		InstructionRun(start=Address(low), code=memory_at(program, Address(low), high - low))
		for low, high in pairwise(chain((code_start,), inner_edges, (code_end,)))
		if high > low
	)


def instruction_set_at(program: Program, address: Address) -> InstructionSet:
	match program.machine:
		case Machine.EM_X86_64:
			return InstructionSet.X86_64
		case Machine.EM_386:
			return InstructionSet.X86_32
		case Machine.EM_ARM:
			index = bisect_right(program.arm_code, address, key=_span_start) - 1
			return (
				InstructionSet.A32
				if index >= 0 and address < program.arm_code[index].end
				else InstructionSet.T32
			)
		case _ as unreachable:
			assert_never(unreachable)


def _span_start(span: Span) -> Address:
	return span.start
