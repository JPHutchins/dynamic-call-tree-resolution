# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The constant argument values GCC's DWARF call-site entries record for an Arm image, and
whether the value-set analysis allows each one at its call."""

from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING, Final, assert_never

from elftools.elf.elffile import ELFFile
from salix import Struct

from dynamic_call_tree_resolution import Address, load
from dynamic_call_tree_resolution.model import aligned
from dynamic_call_tree_resolution.vsa import analyze
from dynamic_call_tree_resolution.vsa.cfg import control_flow_graphs
from dynamic_call_tree_resolution.vsa.lattice import Known, Top

if TYPE_CHECKING:
	from collections.abc import Iterator, Mapping
	from pathlib import Path

	from elftools.dwarf.die import DIE

	from dynamic_call_tree_resolution.vsa.interpret import CallObservation
	from dynamic_call_tree_resolution.vsa.lattice import ValueSet

_CALL_SITES: Final = frozenset({"DW_TAG_call_site", "DW_TAG_GNU_call_site"})
_PARAMETERS: Final = frozenset({"DW_TAG_call_site_parameter", "DW_TAG_GNU_call_site_parameter"})
_DW_OP_ADDR: Final = 0x03
_DW_OP_CONST1U: Final = 0x08
_DW_OP_CONST1S: Final = 0x09
_DW_OP_CONST2U: Final = 0x0A
_DW_OP_CONST2S: Final = 0x0B
_DW_OP_CONST4U: Final = 0x0C
_DW_OP_CONST4S: Final = 0x0D
_DW_OP_CONSTU: Final = 0x10
_DW_OP_CONSTS: Final = 0x11
_DW_OP_LIT0: Final = 0x30
_DW_OP_LIT31: Final = 0x4F
_DW_OP_REG0: Final = 0x50
_DW_OP_REG3: Final = 0x53
_DW_OP_BREG13: Final = 0x7D
_REGISTER_ARGUMENTS: Final = 4
_WORD: Final = 4
_WORD_BITS: Final = 32
_FIXED: Final = {
	_DW_OP_CONST1U: (1, False),
	_DW_OP_CONST1S: (1, True),
	_DW_OP_CONST2U: (2, False),
	_DW_OP_CONST2S: (2, True),
	_DW_OP_CONST4U: (4, False),
	_DW_OP_CONST4S: (4, True),
}


class CallValue(Struct):
	"""A constant DWARF says a call passes in one argument position."""

	return_pc: int
	position: int
	"""r0-r3 are 0-3; the stack word at ``sp + 4k`` is ``4 + k``."""
	value: int


class Checked(Struct):
	"""Each constant DWARF call value, against the arguments the analysis has at that call."""

	agreed: int
	excluded: tuple[CallValue, ...]
	"""Values outside a set the analysis knows: the analysis or GCC is wrong about each."""


def checked(elf: Path) -> Checked:
	program = load(elf)
	call_at = {
		instruction.address + instruction.size: instruction.address
		for _, blocks in control_flow_graphs(program).values()
		for block in blocks
		for instruction in block.instructions
	}
	arguments = _arguments(analyze(program).observations)
	starts = frozenset(aligned(function.address) for function in program.functions.values())
	known = tuple(
		(value, candidates)
		for value in call_values(elf)
		for candidates in (arguments.get((call_at.get(value.return_pc, -1), value.position)),)
		if candidates is not None
	)
	return Checked(
		agreed=sum(_agrees(value.value, candidates, starts) for value, candidates in known),
		excluded=tuple(
			value for value, candidates in known if not _agrees(value.value, candidates, starts)
		),
	)


def _arguments(
	observations: tuple[CallObservation, ...],
) -> Mapping[tuple[int, int], frozenset[Address] | None]:
	"""Each call's known values by argument position; ``None`` where any observation is Top."""
	return {
		key: _joined(tuple(value for _, value in group))
		for key, group in groupby(
			sorted(
				(
					((observation.site, position), value)
					for observation in observations
					for position, value in observation.arguments.items()
				),
				key=_argument_key,
			),
			key=_argument_key,
		)
	}


def _argument_key(argument: tuple[tuple[int, int], ValueSet]) -> tuple[int, int]:
	return argument[0]


def _joined(values: tuple[ValueSet, ...]) -> frozenset[Address] | None:
	known = tuple(found for value in values for found in (_known(value),))
	return (
		None
		if any(found is None for found in known)
		else frozenset(address for found in known if found is not None for address in found)
	)


def _known(value: ValueSet) -> frozenset[Address] | None:
	match value:
		case Known(values=values):
			return values
		case Top():
			return None
		case _ as unreachable:
			assert_never(unreachable)


def _agrees(value: int, candidates: frozenset[Address], starts: frozenset[Address]) -> bool:
	"""Whether a DWARF value is one of the candidates, as GCC writes it.

	GCC types a value by its parameter, so a ``uint8_t`` 0x80 can read ``DW_OP_const1s -128``:
	a negative value also matches its low byte or halfword. A code address may carry the Thumb
	bit on either side.

	>>> _agrees(-128, frozenset({0x80}), frozenset())
	True
	>>> _agrees(-128, frozenset({0xFFFF_FF80}), frozenset())
	True
	>>> _agrees(-128, frozenset({0x180}), frozenset())
	False
	>>> _agrees(0x180, frozenset({0x80}), frozenset())
	False
	>>> _agrees(-0x5EFD_F3FF, frozenset({0xA102_0C01}), frozenset())
	True
	>>> _agrees(0x101, frozenset({0x100}), frozenset({0x100}))
	True
	>>> _agrees(3, frozenset({2}), frozenset())
	False
	"""
	return any(
		candidate == value % (1 << _WORD_BITS)
		or (value < 0 and candidate in (value % (1 << 8), value % (1 << 16)))
		for candidate in candidates
	) or (
		aligned(Address(value)) in starts
		and any(aligned(candidate) == aligned(Address(value)) for candidate in candidates)
	)


def call_values(elf: Path) -> tuple[CallValue, ...]:
	with elf.open("rb") as stream:
		dwarf = ELFFile(stream).get_dwarf_info(relocate_dwarf_sections=False)
		return tuple(
			CallValue(return_pc=return_pc, position=position, value=value)
			for unit in dwarf.iter_CUs()
			for site in _dies(unit.get_top_DIE())
			if site.tag in _CALL_SITES
			for return_pc in (_return_pc(site),)
			if return_pc
			for parameter in site.iter_children()
			if parameter.tag in _PARAMETERS
			for position, value in ((_position(parameter), _value(parameter)),)
			if position is not None and value is not None
		)


def _dies(die: DIE) -> Iterator[DIE]:
	yield die
	for child in die.iter_children():
		yield from _dies(child)


def _return_pc(site: DIE) -> int:
	"""Zero for a call in a function the linker discarded."""
	attribute = site.attributes.get("DW_AT_call_return_pc") or site.attributes.get("DW_AT_low_pc")
	return attribute.value if attribute is not None and isinstance(attribute.value, int) else 0


def _expression(parameter: DIE, *names: str) -> bytes | None:
	attribute = next(
		(parameter.attributes[name] for name in names if name in parameter.attributes), None
	)
	return (
		bytes(attribute.value)
		if attribute is not None and isinstance(attribute.value, list)
		else None
	)


def _position(parameter: DIE) -> int | None:
	expression = _expression(parameter, "DW_AT_location")
	return None if expression is None else _position_of(expression)


def _position_of(expression: bytes) -> int | None:
	"""The argument position a single-operation location names.

	>>> _position_of(bytes([0x52]))
	2
	>>> _position_of(bytes([0x7D, 0x08]))
	6
	>>> _position_of(bytes([0x55])) is None
	True
	"""
	match tuple(expression):
		case (op,) if _DW_OP_REG0 <= op <= _DW_OP_REG3:
			return op - _DW_OP_REG0
		case (op, *rest) if op == _DW_OP_BREG13:
			offset, end = _leb128(bytes(rest), signed=True)
			return (
				_REGISTER_ARGUMENTS + offset // _WORD
				if end == len(rest) and offset >= 0 and offset % _WORD == 0
				else None
			)
		case _:
			return None


def _value(parameter: DIE) -> int | None:
	expression = _expression(parameter, "DW_AT_call_value", "DW_AT_GNU_call_site_value")
	return None if expression is None else _constant(expression)


def _constant(expression: bytes) -> int | None:
	"""The value of a single-operation constant expression.

	>>> _constant(bytes([0x32]))
	2
	>>> _constant(bytes([0x09, 0x80]))
	-128
	>>> _constant(bytes([0x03, 0x2D, 0x4D, 0x00, 0x00]))
	19757
	>>> _constant(bytes([0x11, 0x7F]))
	-1
	>>> _constant(bytes([0x75, 0x00])) is None
	True
	"""
	match tuple(expression):
		case (op,) if _DW_OP_LIT0 <= op <= _DW_OP_LIT31:
			return op - _DW_OP_LIT0
		case (op, *operand) if op == _DW_OP_ADDR and len(operand) == _WORD:
			return int.from_bytes(bytes(operand), "little")
		case (op, *operand) if op in _FIXED and len(operand) == _FIXED[op][0]:
			return int.from_bytes(bytes(operand), "little", signed=_FIXED[op][1])
		case (op, *operand) if op in (_DW_OP_CONSTU, _DW_OP_CONSTS):
			value, end = _leb128(bytes(operand), signed=op == _DW_OP_CONSTS)
			return value if end == len(operand) else None
		case _:
			return None


def _leb128(data: bytes, *, signed: bool) -> tuple[int, int]:
	"""A LEB128 number and how many bytes it takes.

	>>> _leb128(bytes([0xE5, 0x8E, 0x26]), signed=False)
	(624485, 3)
	>>> _leb128(bytes([0x7F]), signed=True)
	(-1, 1)
	"""
	end = next((index + 1 for index, byte in enumerate(data) if not byte & 0x80), len(data))
	value = sum((byte & 0x7F) << (7 * index) for index, byte in enumerate(data[:end]))
	return (
		value - (1 << (7 * end)) if signed and end and data[end - 1] & 0x40 else value,
		end,
	)
