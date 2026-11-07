# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""What each store a dump shows writes, and what each call hands code without a dump (#341)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final, assert_never

from salix import Struct

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
	indexed_array,
	is_ssa,
	is_variable,
	member_record,
	named_member,
	record_layout,
	split_member,
	type_record,
	variable_symbol,
)
from dynamic_call_tree_resolution.gcc_dumps.records import Located, located, location_fields
from dynamic_call_tree_resolution.gcc_dumps.units import Record, anonymous_name
from dynamic_call_tree_resolution.model import Field

if TYPE_CHECKING:
	from pathlib import Path

	from dynamic_call_tree_resolution.descriptors import WriteTarget
	from dynamic_call_tree_resolution.gcc_dumps.dump import DumpFunction
	from dynamic_call_tree_resolution.gcc_dumps.expressions import Member, Scope
	from dynamic_call_tree_resolution.gcc_dumps.units import UnitTypes

EVERY_STORE: Final = "writes\tcomplete"
INDIRECT: Final = "<indirect>"
ASSEMBLY: Final = "<asm>"
_STORED_RESULT: Final = re.compile(r"^(.+?) = \S+ \(")
_CONDITION: Final = re.compile(r"^if \(")
_CLOBBER: Final = re.compile(r"^.+? = \{CLOBBER")
_ATOMIC_READ: Final = re.compile(r"^__atomic_load(?:_\d+)?$")
_FILL: Final = re.compile(r"^(?:__builtin_)?(?:memcpy|memmove|memset|bzero)$")
_BUILTIN: Final = re.compile(r"^(?:__(?:builtin|atomic|sync)_|\.)")
_MEMORY: Final = re.compile(
	r"^MEM(?: <(.+?)>)? ?\[\((.+?) \*(?: \{ref-all\})?\)([^\[\]]+?)(?: \+ (\d+)B)?\]$"
)
_DEREFERENCE: Final = re.compile(r"^\(\*(.+)\)$|^\*(.+)$")
_ASSEMBLY: Final = re.compile(r"^__asm__(?: __volatile__)?\(\"(?:[^\"\\]|\\.)*\"(.*)\)$")
_OPERAND: Final = re.compile(r"^\"([^\"]*)\" (.+)$")
_OPERAND_LISTS: Final = re.compile(r"\s*:\s*")
_MEMORY_CONSTRAINT: Final = re.compile(r"[mQ]")
_WRAP: Final = 1 << 32
"""GCC prints a negative ``MEM`` offset as its 32-bit unsigned wrap."""


class WriteRecord(Struct):
	"""A store a statement makes, or a builtin's write through an argument."""

	at: Located
	target: WriteTarget


class PassRecord(Struct):
	"""A pointer a call hands a callee without a dump, an unknown callee, or inline assembly."""

	at: Located
	callee: str
	"""The callee's name, ``<indirect>`` or ``<asm>``."""
	argument: int
	target: WriteTarget


type WriteOrPass = WriteRecord | PassRecord


def unit_writes(
	unit: str,
	functions: tuple[DumpFunction, ...],
	types: UnitTypes,
	root: Path,
	dumped: frozenset[str],
) -> tuple[WriteOrPass, ...]:
	"""Every function's writes, with ``dumped`` the functions some dump of the build defines."""
	return tuple(
		record
		for function in functions
		for scope in (function_scope(types, function),)
		for statement in function.statements
		for record in _writes(
			scope, dumped, located(unit, function, statement, root), statement.body
		)
	)


def write_line(record: WriteOrPass) -> str:
	match record:
		case WriteRecord(at=at, target=target):
			return "\t".join(("write", *location_fields(at), *write_target_fields(target)))
		case PassRecord(at=at, callee=callee, argument=argument, target=target):
			return "\t".join(
				("pass", *location_fields(at), callee, str(argument), *write_target_fields(target))
			)
		case _ as unreachable:
			assert_never(unreachable)


def written(scope: Scope, expression: str) -> WriteTarget:
	"""What a store's left side names: a field, a variable, every member of a record, or nothing."""
	member = split_member(expression)
	array = indexed_array(expression)
	memory = _MEMORY.match(expression)
	dereference = _DEREFERENCE.match(expression)
	if member is not None:
		return _field(scope, named_member(member))
	if array is not None:
		return written(scope, array)
	if memory is not None:
		return _memory(scope, memory.group(2), memory.group(3), int(memory.group(4) or 0))
	if dereference is not None:
		return _pointed(scope, dereference.group(1) or dereference.group(2))
	if expression.startswith('"'):
		return Unnamed()
	if expression == "<retval>":
		return _pointed_record(type_record(scope, scope.function.result))
	if expression in scope.function.parameters or is_variable(scope, expression):
		return _object(scope, expression)
	return Unknown()


def _writes(
	scope: Scope, dumped: frozenset[str], at: Located, body: str
) -> tuple[WriteOrPass, ...]:
	call = CALL.match(body)
	assignment = ASSIGNMENT.match(body)
	if body.startswith("__asm__"):
		return _assembly(scope, at, _ASSEMBLY.match(body))
	if call is not None and _CONDITION.match(body) is None:
		result = _STORED_RESULT.match(body)
		return (
			*(_stored(scope, at, result.group(1)) if result is not None else ()),
			*_called(scope, dumped, at, call.group(1), _arguments(call.group(2))),
		)
	if assignment is not None and _CLOBBER.match(body) is None:
		return _stored(scope, at, assignment.group(1))
	return ()


def _stored(scope: Scope, at: Located, target: str) -> tuple[WriteRecord, ...]:
	return () if is_ssa(scope, target) else (WriteRecord(at=at, target=written(scope, target)),)


def _called(
	scope: Scope, dumped: frozenset[str], at: Located, callee: str, arguments: tuple[str, ...]
) -> tuple[WriteOrPass, ...]:
	"""A builtin's writes through its arguments, and the pointers a call hands code without a dump."""
	indirect = is_ssa(scope, callee) and callee not in scope.types.functions
	handed = indirect or (callee not in dumped and _BUILTIN.match(callee) is None)
	return (
		*(
			WriteRecord(at=at, target=_pointed(scope, arguments[index]))
			for index in _written_through(scope, callee, arguments)
		),
		*(
			PassRecord(
				at=at,
				callee=INDIRECT if indirect else callee,
				argument=index,
				target=_pointed(scope, argument),
			)
			for index, argument in enumerate(arguments)
			if handed and _pointer(scope, argument)
		),
	)


def _arguments(text: str) -> tuple[str, ...]:
	return tuple(argument for argument in top_level_parts(text) if argument)


def _written_through(scope: Scope, callee: str, arguments: tuple[str, ...]) -> tuple[int, ...]:
	"""The arguments a builtin or GCC's internal function writes through.

	Every pointer, except that an atomic load writes nothing and a fill only its destination.
	"""
	if _ATOMIC_READ.match(callee):
		return ()
	if _FILL.match(callee):
		return (0,)
	if _BUILTIN.match(callee):
		return tuple(index for index, argument in enumerate(arguments) if _pointer(scope, argument))
	return ()


def _assembly(scope: Scope, at: Located, assembly: re.Match[str] | None) -> tuple[WriteOrPass, ...]:
	"""Inline assembly writes its memory outputs, and is handed every pointer among its operands.

	Its operands are unreadable when the template spans lines: then it may write anything.
	"""
	if assembly is None:
		return (WriteRecord(at=at, target=Unknown()),)
	lists = _OPERAND_LISTS.split(assembly.group(1))[1:]
	outputs = _operands(lists[0]) if lists else ()
	inputs = _operands(lists[1]) if len(lists) > 1 else ()
	operands = (*outputs, *inputs)
	return (
		*(
			WriteRecord(at=at, target=written(scope, operand.group(2)))
			for operand in outputs
			if operand is not None and _MEMORY_CONSTRAINT.search(operand.group(1)) is not None
		),
		*(
			PassRecord(
				at=at, callee=ASSEMBLY, argument=index, target=_pointed(scope, operand.group(2))
			)
			for index, operand in enumerate(operands)
			if operand is not None and _pointer(scope, operand.group(2))
		),
		*((WriteRecord(at=at, target=Unknown()),) if None in operands else ()),
	)


def _operands(text: str) -> tuple[re.Match[str] | None, ...]:
	return tuple(_OPERAND.match(operand) for operand in top_level_parts(text) if operand)


def _field(scope: Scope, member: Member) -> WriteTarget:
	"""A member's field, or, where the types can't name its record, what it is a member of."""
	record = member_record(scope, member)
	return (
		Field(record=record.name, member=member.member)
		if record is not None
		else written(scope, member.base)
	)


def _memory(scope: Scope, cast: str, base: str, offset: int) -> WriteTarget:
	"""A ``MEM`` store: of the object its address names, else at an offset into its pointee's record."""
	if base.startswith("&"):
		return written(scope, base[1:])
	record = _pointee(scope, base) or type_record(scope, cast)
	return (
		_at_offset(scope, record, offset - _WRAP if offset >= _WRAP // 2 else offset)
		if record is not None
		else Unnamed()
	)


def _at_offset(scope: Scope, record: Record, offset: int) -> WriteTarget:
	"""The innermost named member at the offset, in whichever element it falls; else every member."""
	layout = record_layout(scope, record)
	if layout is None or layout.size == 0:
		return EveryMember(record=record.name)
	within = offset % layout.size
	match tuple(
		(name, extent)
		for name, extent in layout.extents.items()
		if extent.size is not None and extent.offset <= within < extent.offset + extent.size
	):
		case ((name, extent),) if extent.layout is not None:
			inner = _at_offset(
				scope,
				Record(name=anonymous_name(extent.layout), key=extent.layout),
				within - extent.offset,
			)
			return (
				Field(record=record.name, member=name) if isinstance(inner, EveryMember) else inner
			)
		case ((name, _),):
			return Field(record=record.name, member=name)
		case _:
			return EveryMember(record=record.name)


def _pointed(scope: Scope, pointer: str) -> WriteTarget:
	"""What a write through a pointer names: the object of ``&x``, else the pointee's record.

	A pointer defined as ``&x`` names ``x`` too.
	"""
	address = scope.definitions.get(pointer, pointer)
	if address.startswith("&"):
		return written(scope, address[1:])
	return _pointed_record(_pointee(scope, pointer))


def _pointed_record(record: Record | None) -> WriteTarget:
	return EveryMember(record=record.name) if record is not None else Unnamed()


def _pointee(scope: Scope, pointer: str) -> Record | None:
	"""The record a pointer points to: none for a pointer to a pointer, which writes no member."""
	declared = declared_type(scope, pointer)
	return (
		type_record(scope, declared) if declared is not None and declared.count("*") == 1 else None
	)


def _object(scope: Scope, name: str) -> WriteTarget:
	"""A named object: a file-scope or function-scope static variable, or a local or parameter."""
	declared = scope.function.declarations.get(name)
	if name not in scope.function.parameters and (
		declared is None or declared.startswith("static ")
	):
		return Variable(symbol=variable_symbol(scope, name))
	held = declared_type(scope, name)
	record = type_record(scope, held) if held is not None and "*" not in held else None
	return EveryMember(record=record.name) if record is not None else Unnamed()


def _pointer(scope: Scope, argument: str) -> bool:
	declared = declared_type(scope, argument)
	return argument.startswith("&") or (declared is not None and "*" in declared)
