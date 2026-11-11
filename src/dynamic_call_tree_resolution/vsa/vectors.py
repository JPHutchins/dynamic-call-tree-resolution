# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The exception vector table an M-profile core takes its handlers from."""

from __future__ import annotations

from functools import partial
from itertools import takewhile
from typing import TYPE_CHECKING, Final, assert_never

from salix import Struct

from dynamic_call_tree_resolution.composition import flatmap
from dynamic_call_tree_resolution.model import Address, ArmCore, ArmProfile
from dynamic_call_tree_resolution.vsa.fallback import referenced_only_at, referrers

if TYPE_CHECKING:
	from collections.abc import Iterator, Mapping

	from dynamic_call_tree_resolution.model import Program, Section

_STACK_ALIGNMENT: Final = 8
RESET: Final = 1
NMI: Final = 2
HARD_FAULT: Final = 3


def vector_table(program: Program) -> Mapping[Address, Address]:
	"""Each handler slot of the vector table, with its handler."""
	match program.arm_core:
		case ArmCore(profile=ArmProfile.MICROCONTROLLER) if program.entry_point != 0:
			tables = flatmap(partial(_section_tables, program), sorted(program.sections))
			return next(tables, dict[Address, Address]())
		case ArmCore() | None:
			return {}
		case _ as unreachable:
			assert_never(unreachable)


def exception_handlers(program: Program) -> Mapping[int, Address]:
	"""Each exception the vector table handles, by exception number, with its handler."""
	table = vector_table(program)
	reset = min(table, default=0)
	return {
		(slot - reset) // program.pointer_size + RESET: handler for slot, handler in table.items()
	}


def hardware_handlers(program: Program) -> frozenset[Address]:
	"""The handlers whose address the image holds only in the vector table."""
	table = vector_table(program)
	slots_by_handler = {
		handler: frozenset(slot for slot, held in table.items() if held == handler)
		for handler in frozenset(table.values())
	}
	return referenced_only_at(referrers(program, frozenset(slots_by_handler)), slots_by_handler)


def _section_tables(program: Program, base: Address) -> Iterator[dict[Address, Address]]:
	section = program.sections[base]
	return (
		_table(program, base, section, offset)
		for offset in range(program.pointer_size, len(section.data), program.pointer_size)
		if _word(program, section, offset) == program.entry_point
		and _is_initial_stack(_word(program, section, offset - program.pointer_size))
	)


def _table(program: Program, base: Address, section: Section, start: int) -> dict[Address, Address]:
	return {
		entry.slot: entry.word
		for entry in takewhile(
			partial(_handler_or_reserved, program), _words(program, base, section, start)
		)
		if entry.word != 0
	}


class _VectorWord(Struct):
	"""A word of a vector table, with the slot that holds it."""

	slot: Address
	word: Address


def _words(program: Program, base: Address, section: Section, start: int) -> Iterator[_VectorWord]:
	return (
		_VectorWord(slot=Address(base + offset), word=_word(program, section, offset))
		for offset in range(
			start, len(section.data) - program.pointer_size + 1, program.pointer_size
		)
	)


def _word(program: Program, section: Section, offset: int) -> Address:
	return Address(
		int.from_bytes(section.data[offset : offset + program.pointer_size], program.byte_order)
	)


def _handler_or_reserved(program: Program, entry: _VectorWord) -> bool:
	return entry.word == 0 or entry.word in program.functions


def _is_initial_stack(value: int) -> bool:
	return value != 0 and value % _STACK_ALIGNMENT == 0
