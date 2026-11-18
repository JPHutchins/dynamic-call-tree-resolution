# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The scalar fields and variables a build takes the address of (#341).

A store that names no field can reach only these.
"""

from __future__ import annotations

import re
from functools import partial
from itertools import accumulate, chain, pairwise
from typing import TYPE_CHECKING, Final, assert_never

from dynamic_call_tree_resolution.composition import flatmap, unanimous
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
	function_scope,
	is_static,
	possible_types,
	record_layout,
	type_record,
)
from dynamic_call_tree_resolution.gcc_dumps.writes import (
	assembly_operands,
	pointed_member,
	pointee,
	written,
)
from dynamic_call_tree_resolution.model import Field

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator

	from dynamic_call_tree_resolution.descriptors import WriteTarget
	from dynamic_call_tree_resolution.gcc_dumps.dump import DumpFunction, Statement
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


def unit_taken(functions: Iterable[DumpFunction], types: UnitTypes) -> Iterator[WriteTarget]:
	"""Each scalar field or variable whose address a function uses as a value.

	The value may be assigned, passed, returned, an asm operand, or joined by a PHI.
	"""
	return flatmap(partial(_function_taken, types), functions)


def _function_taken(types: UnitTypes, function: DumpFunction) -> Iterator[WriteTarget]:
	scope = function_scope(types, function)
	return chain(_addressed(scope, function), _interior(scope, function), _unreadable(function))


def _unreadable(function: DumpFunction) -> Iterator[WriteTarget]:
	"""Inline assembly whose operands can't be read may take any address."""
	return (
		Unknown()
		for statement in function.statements
		if statement.body.startswith("__asm__") and assembly_operands(statement.body) is None
	)


def _addressed(scope: Scope, function: DumpFunction) -> Iterator[WriteTarget]:
	return flatmap(partial(_addressed_target, scope), _values(function))


def _addressed_target(scope: Scope, value: str) -> tuple[WriteTarget, ...]:
	address = _CAST.sub("", value)
	if not address.startswith("&"):
		return ()
	target = written(scope, address[1:])
	return (target,) if _scalar(scope, target, address[1:]) else ()


def _interior(scope: Scope, function: DumpFunction) -> Iterator[WriteTarget]:
	"""What pointer arithmetic on a record pointer reaches, when its result points to no record.

	A variable offset reaches every member of the record.
	"""
	return flatmap(partial(_interior_target, scope), function.statements)


def _interior_target(scope: Scope, statement: Statement) -> tuple[WriteTarget, ...]:
	assignment = ASSIGNMENT.match(statement.body)
	if assignment is None or _points_to_record(scope, assignment.group(1)):
		return ()
	arithmetic = _ARITHMETIC.match(assignment.group(2))
	if arithmetic is None:
		return ()
	target = _reached(scope, arithmetic.group(1), arithmetic.group(2))
	return (target,) if target is not None else ()


def _reached(scope: Scope, pointer: str, offset: str) -> WriteTarget | None:
	"""The member a constant byte offset from a pointer reaches, or every member for a variable one."""
	return (
		pointed_member(scope, pointer, int(offset), 1)
		if offset.isdigit()
		else pointed_member(scope, pointer, 0, None)
	)


def _points_to_record(scope: Scope, name: str) -> bool:
	return unanimous(
		(pointee(scope, declared) is not None for declared in possible_types(scope, name)), False
	)


def _local(scope: Scope, name: str) -> bool:
	return not is_static(scope, name) and (
		name in scope.function.parameters or name in scope.function.declarations
	)


def taken_line(target: WriteTarget) -> str:
	return "\t".join(("taken", *write_target_fields(target)))


def _values(function: DumpFunction) -> Iterator[str]:
	return chain(
		flatmap(lambda statement: _statement_values(statement.body), function.statements),
		flatmap(_phi_values, function.phis),
	)


def _phi_values(phi: str) -> Iterable[str]:
	arguments = _PHI_ARGUMENTS.search(phi)
	return (
		(_PREDECESSOR.sub("", argument) for argument in top_level_parts(arguments.group(1)))
		if arguments is not None
		else ()
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
	edges = flatmap(lambda operator: (operator.start(), operator.end()), cuts)
	return tuple(
		expression[start:end].strip()
		for start, end in pairwise(chain((0,), edges, (len(expression),)))
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
			return variable is None or any(
				declared is None or "*" in declared or type_record(scope, declared) is None
				for declared in possible_types(scope, variable.group())
			)
		case Unknown():
			return not _local(scope, expression)
		case EveryMember() | Unnamed():
			return False
		case _ as unreachable:
			assert_never(unreachable)
