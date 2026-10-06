# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The exception vector table an M-profile core takes its handlers from."""

from __future__ import annotations

from functools import partial
from itertools import takewhile
from typing import TYPE_CHECKING, Final, assert_never

from dynamic_call_tree_resolution.model import Address, ArmCore, ArmProfile
from dynamic_call_tree_resolution.vsa.fallback import referenced_only_at, referrers

if TYPE_CHECKING:
	from collections.abc import Mapping

	from dynamic_call_tree_resolution.model import Program, Section

_STACK_ALIGNMENT: Final = 8


def vector_table(program: Program) -> Mapping[Address, Address]:
	"""Each handler slot of the vector table, with its handler."""
	match program.arm_core:
		case ArmCore(profile=ArmProfile.MICROCONTROLLER) if program.entry_point != 0:
			return next(
				(
					{
						slot: handler
						for slot, handler in takewhile(
							partial(_handler_or_reserved, program),
							_words(program, base, section, offset),
						)
						if handler != 0
					}
					for base, section in sorted(program.sections.items())
					for offset in range(
						program.pointer_size, len(section.data), program.pointer_size
					)
					if _word(program, section, offset) == program.entry_point
					and _is_initial_stack(_word(program, section, offset - program.pointer_size))
				),
				dict[Address, Address](),
			)
		case ArmCore() | None:
			return {}
		case _ as unreachable:
			assert_never(unreachable)


def hardware_handlers(program: Program) -> frozenset[Address]:
	"""The handlers whose address the image holds only in the vector table."""
	slots_by_handler = {
		handler: frozenset(slot for slot, held in table.items() if held == handler)
		for table in (vector_table(program),)
		for handler in frozenset(table.values())
	}
	return referenced_only_at(referrers(program, frozenset(slots_by_handler)), slots_by_handler)


def _words(
	program: Program, base: Address, section: Section, start: int
) -> tuple[tuple[Address, Address], ...]:
	return tuple(
		(Address(base + offset), _word(program, section, offset))
		for offset in range(
			start, len(section.data) - program.pointer_size + 1, program.pointer_size
		)
	)


def _word(program: Program, section: Section, offset: int) -> Address:
	return Address(
		int.from_bytes(section.data[offset : offset + program.pointer_size], program.byte_order)
	)


def _handler_or_reserved(program: Program, word: tuple[Address, Address]) -> bool:
	return word[1] == 0 or word[1] in program.functions


def _is_initial_stack(value: int) -> bool:
	return value != 0 and value % _STACK_ALIGNMENT == 0
