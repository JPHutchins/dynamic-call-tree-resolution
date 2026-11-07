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
from dynamic_call_tree_resolution.gcc_dumps.writes import written
from dynamic_call_tree_resolution.model import Field

if TYPE_CHECKING:
	from dynamic_call_tree_resolution.descriptors import WriteTarget
	from dynamic_call_tree_resolution.gcc_dumps.dump import DumpFunction
	from dynamic_call_tree_resolution.gcc_dumps.expressions import Scope
	from dynamic_call_tree_resolution.gcc_dumps.units import UnitTypes

_RETURN: Final = re.compile(r"^return (.+)$")
_CONDITION: Final = re.compile(r"^if \(")
_OPERAND: Final = re.compile(r"\"[^\"]*\" ([^,:)]*[^,:)\s])")
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
		for value in _values(function)
		if value.startswith("&")
		for target in (written(scope, value[1:]),)
		if _scalar(scope, target, value[1:])
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
		return tuple(operand.group(1) for operand in _OPERAND.finditer(body))
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
