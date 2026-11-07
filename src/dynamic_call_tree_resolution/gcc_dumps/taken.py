# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The scalar fields and variables a build takes the address of (#341).

A store that names no field can reach only these.
"""

from __future__ import annotations

import re
from itertools import accumulate, pairwise
from typing import TYPE_CHECKING, Final, assert_never

from dynamic_call_tree_resolution.descriptors import (
	EveryMember,
	Unknown,
	Unnamed,
	Variable,
	write_target_fields,
)
from dynamic_call_tree_resolution.gcc_dumps.dump import top_level_parts
from dynamic_call_tree_resolution.gcc_dumps.expressions import (
	ASSIGNMENT,
	CALL,
	declared_type,
	function_scope,
	record_layout,
	type_record,
)
from dynamic_call_tree_resolution.gcc_dumps.writes import assembly_operands, pointed_member, written
from dynamic_call_tree_resolution.model import Field

if TYPE_CHECKING:
	from dynamic_call_tree_resolution.descriptors import WriteTarget
	from dynamic_call_tree_resolution.gcc_dumps.dump import DumpFunction
	from dynamic_call_tree_resolution.gcc_dumps.expressions import Scope
	from dynamic_call_tree_resolution.gcc_dumps.units import UnitTypes

_RETURN: Final = re.compile(r"^return (.+)$")
_ARITHMETIC: Final = re.compile(r"^(\S+) \+ (\S+)$")
_CAST: Final = re.compile(r"^\((?:[^()]|\([^()]*\))*\) ")
_CONDITION: Final = re.compile(r"^if \(")
_PHI_ARGUMENTS: Final = re.compile(r"PHI <(.*)>$")
_PREDECESSOR: Final = re.compile(r"\(\d+\)$")
_VARIABLE: Final = re.compile(r"^[A-Za-z_][\w.$]*")
_MEMORY_TYPE: Final = re.compile(r"<[^<>]*>")
_BINARY: Final = re.compile(r" (?:[-+*/%^|&<>]|[<>!=]=|<<|>>) ")


def unit_taken(functions: tuple[DumpFunction, ...], types: UnitTypes) -> tuple[WriteTarget, ...]:
	"""Each scalar field or variable whose address a function uses as a value.

	The value may be assigned, passed, returned, an asm operand, or joined by a PHI.
	"""
	return tuple(
		target
		for function in functions
		for scope in (function_scope(types, function),)
		for target in (
			*_addressed(scope, function),
			*_interior(scope, function),
			*_unreadable(function),
		)
	)


def _unreadable(function: DumpFunction) -> tuple[WriteTarget, ...]:
	"""Inline assembly whose operands can't be read may take any address."""
	return tuple(
		Unknown()
		for statement in function.statements
		if statement.body.startswith("__asm__") and assembly_operands(statement.body) is None
	)


def _addressed(scope: Scope, function: DumpFunction) -> tuple[WriteTarget, ...]:
	return tuple(
		target
		for value in _values(function)
		for address in (_CAST.sub("", value),)
		if address.startswith("&")
		for target in (written(scope, address[1:]),)
		if _scalar(scope, target, address[1:])
	)


def _interior(scope: Scope, function: DumpFunction) -> tuple[WriteTarget, ...]:
	"""What pointer arithmetic on a record pointer reaches, when its result points to no record.

	A variable offset reaches every member of the record.
	"""
	return tuple(
		target
		for statement in function.statements
		for assignment in (ASSIGNMENT.match(statement.body),)
		if assignment is not None and not _points_to_record(scope, assignment.group(1))
		for arithmetic in (_ARITHMETIC.match(assignment.group(2)),)
		if arithmetic is not None
		for target in (_reached(scope, arithmetic.group(1), arithmetic.group(2)),)
		if target is not None
	)


def _reached(scope: Scope, pointer: str, offset: str) -> WriteTarget | None:
	"""The member a constant byte offset from a pointer reaches, or every member for a variable one."""
	return (
		pointed_member(scope, pointer, int(offset), 1)
		if offset.isdigit()
		else pointed_member(scope, pointer, 0, None)
	)


def _points_to_record(scope: Scope, name: str) -> bool:
	declared = declared_type(scope, name)
	return (
		declared is not None
		and declared.count("*") == 1
		and type_record(scope, declared) is not None
	)


def taken_line(target: WriteTarget) -> str:
	return "\t".join(("taken", *write_target_fields(target)))


def _values(function: DumpFunction) -> tuple[str, ...]:
	return (
		*(
			value
			for statement in function.statements
			for value in _statement_values(statement.body)
		),
		*(
			_PREDECESSOR.sub("", argument)
			for phi in function.phis
			for arguments in (_PHI_ARGUMENTS.search(phi),)
			if arguments is not None
			for argument in top_level_parts(arguments.group(1))
		),
	)


def _statement_values(body: str) -> tuple[str, ...]:
	call = CALL.match(body)
	assignment = ASSIGNMENT.match(body)
	returned = _RETURN.match(body)
	if body.startswith("__asm__"):
		return assembly_operands(body) or ()
	if call is not None and _CONDITION.match(body) is not None:
		return _operands(call.group(2))
	if call is not None:
		return top_level_parts(call.group(2))
	if assignment is not None:
		return _operands(assignment.group(2))
	return _operands(returned.group(1)) if returned is not None else ()


def _operands(expression: str) -> tuple[str, ...]:
	"""The operands of an expression's top-level operator, or the expression itself.

	>>> _operands("&end - &start"), _operands("&MEM <char[2]> [(void *)&buf + 2B] - p_1")
	(('&end', '&start'), ('&MEM <char[2]> [(void *)&buf + 2B]', 'p_1'))
	"""
	masked = _MEMORY_TYPE.sub(_blanked, expression)
	depths = tuple(accumulate((character in "([") - (character in ")]") for character in masked))
	cuts = tuple(operator for operator in _BINARY.finditer(masked) if depths[operator.start()] == 0)
	return tuple(
		expression[start:end].strip()
		for start, end in pairwise(
			(
				0,
				*(edge for operator in cuts for edge in (operator.start(), operator.end())),
				len(expression),
			)
		)
	)[::2]


def _blanked(memory_type: re.Match[str]) -> str:
	return " " * len(memory_type.group())


def _scalar(scope: Scope, target: WriteTarget, expression: str) -> bool:
	"""Whether a write through the address may name no field: it is not of a record."""
	match target:
		case Field(record=record, member=member):
			layout = record_layout(scope, type_record(scope, record))
			extent = layout.extents.get(member) if layout is not None else None
			return extent is None or extent.layout is None
		case Variable():
			variable = _VARIABLE.match(expression)
			declared = declared_type(scope, variable.group()) if variable is not None else None
			return declared is None or "*" in declared or type_record(scope, declared) is None
		case Unknown():
			return True
		case EveryMember() | Unnamed():
			return False
		case _ as unreachable:
			assert_never(unreachable)
