# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""What each store a dump shows writes, and what each call hands code without a dump (#341)."""

from __future__ import annotations

import re
from functools import partial
from typing import TYPE_CHECKING, Final, assert_never

from salix import Struct

from dynamic_call_tree_resolution.composition import flatmap
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
	key_record,
	member_record,
	named_member,
	record_layout,
	split_member,
	type_record,
	variable_symbol,
)
from dynamic_call_tree_resolution.gcc_dumps.records import Located, located, location_fields
from dynamic_call_tree_resolution.model import Field

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator
	from pathlib import Path

	from dynamic_call_tree_resolution.descriptors import WriteTarget
	from dynamic_call_tree_resolution.gcc_dumps.dump import DumpFunction
	from dynamic_call_tree_resolution.gcc_dumps.expressions import Member, Scope
	from dynamic_call_tree_resolution.gcc_dumps.units import Record, UnitTypes

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
_ASSEMBLY: Final = re.compile(r"^__asm__(?: __volatile__)?\(\"((?:[^\"\\]|\\.)*)\"(.*)\)$")
_OPERAND: Final = re.compile(r"^\"([^\"]*)\" (.+)$")
_OPERAND_LISTS: Final = re.compile(r"\s*:\s*")
_QUALIFIER: Final = re.compile(r"\b(?:const|volatile)\b ?")
_ARRAY_TYPE: Final = re.compile(r"^(.+)\[(\d+)\]$")
_POINTER_WIDTH: Final = 4
_SCALAR_WIDTHS: Final = {
	"_Bool": 1,
	"char": 1,
	"signed char": 1,
	"unsigned char": 1,
	"short int": 2,
	"short unsigned int": 2,
	"int": 4,
	"unsigned int": 4,
	"long int": 4,
	"long unsigned int": 4,
	"long long int": 8,
	"long long unsigned int": 8,
	"float": 4,
	"double": 8,
	"long double": 8,
}
_NON_STORING: Final = frozenset(
	{"", "bkpt", "cpsid", "cpsie", "dmb", "dsb", "isb", "mov", "movs", "mrs", "msr", "nop", "sev"}
	| {"svc", "wfe", "wfi"}
)
"""The instructions an asm template may hold without storing: barriers, special registers, moves,
hints and the supervisor call, whose handler is code without a dump."""
_WRAP: Final = 1 << 32
_ARRAY_POINTEE: Final = re.compile(r"^.+\[\d+\]\s*\*$")
"""GCC prints a negative ``MEM`` offset as its 32-bit unsigned wrap."""


class WriteRecord(Struct):
	"""A ``descriptors.Write`` as one unit's dump gives it."""

	at: Located
	target: WriteTarget


class PassRecord(Struct):
	"""A ``descriptors.Pass`` as one unit's dump gives it."""

	at: Located
	callee: str
	"""The callee's name, ``<indirect>`` or ``<asm>``."""
	argument: int
	target: WriteTarget


type WriteOrPass = WriteRecord | PassRecord


class _Access(Struct):
	"""A ``MEM`` access: the type it accesses, its address's cast, base and byte offset."""

	type: str
	cast: str
	base: str
	offset: int


class _Span(Struct):
	"""Bytes at an offset into a record, of a width the dump's types may not tell."""

	offset: int
	width: int | None


def unit_writes(
	unit: str,
	functions: Iterable[DumpFunction],
	types: UnitTypes,
	root: Path,
	dumped: frozenset[str],
) -> Iterator[WriteOrPass]:
	"""Every function's writes, with ``dumped`` the functions some dump of the build defines."""
	return flatmap(partial(_function_writes, unit, types, root, dumped), functions)


def _function_writes(
	unit: str, types: UnitTypes, root: Path, dumped: frozenset[str], function: DumpFunction
) -> Iterator[WriteOrPass]:
	scope = function_scope(types, function)
	return flatmap(
		lambda statement: _writes(
			scope, dumped, located(unit, function, statement, root), statement.body
		),
		function.statements,
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
		return _memory(
			scope,
			_Access(
				type=memory.group(1) or memory.group(2),
				cast=memory.group(2),
				base=memory.group(3),
				offset=int(memory.group(4) or 0),
			),
		)
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
	"""Inline assembly writes its outputs, and is handed every pointer among its operands.

	With a ``"memory"`` clobber and an instruction that may store, or operands it can't read, it may
	write anything.
	"""
	if assembly is None:
		return (WriteRecord(at=at, target=Unknown()),)
	lists = _OPERAND_LISTS.split(assembly.group(2))[1:]
	outputs = _operands(lists[0]) if lists else ()
	inputs = _operands(lists[1]) if len(lists) > 1 else ()
	clobbers = lists[2] if len(lists) > 2 else ""
	operands = (*outputs, *inputs)
	return (
		*(
			WriteRecord(at=at, target=written(scope, operand.group(2)))
			for operand in outputs
			if operand is not None and not is_ssa(scope, operand.group(2))
		),
		*(
			PassRecord(
				at=at, callee=ASSEMBLY, argument=index, target=_pointed(scope, operand.group(2))
			)
			for index, operand in enumerate(operands)
			if operand is not None and _pointer(scope, operand.group(2))
		),
		*(
			(WriteRecord(at=at, target=Unknown()),)
			if None in operands or ('"memory"' in clobbers and _may_store(assembly.group(1)))
			else ()
		),
	)


def _may_store(template: str) -> bool:
	r"""Whether an asm template holds an instruction other than those that store nothing.

	>>> _may_store(""), _may_store("MSR basepri, %0"), _may_store("mov r0, %0\n\tsvc 3")
	(False, False, False)
	>>> _may_store("str %1, [%0]")
	True
	"""
	lines = template.replace("\\n", "\n").replace(";", "\n").splitlines()
	return any(
		mnemonic.lower() not in _NON_STORING
		for mnemonic in flatmap(lambda line: line.split()[:1], lines)
		if not mnemonic.startswith(".")
	)


def assembly_operands(body: str) -> tuple[str, ...] | None:
	"""The expressions of an inline asm statement's outputs and inputs; none when unreadable."""
	assembly = _ASSEMBLY.match(body)
	if assembly is None:
		return None
	operands = tuple(flatmap(_operands, _OPERAND_LISTS.split(assembly.group(2))[1:3]))
	return (
		None
		if None in operands
		else tuple(operand.group(2) for operand in operands if operand is not None)
	)


def _operands(text: str) -> tuple[re.Match[str] | None, ...]:
	return tuple(_OPERAND.match(operand) for operand in top_level_parts(text) if operand)


def _field(scope: Scope, member: Member) -> WriteTarget:
	"""A member's field, or, where the types can't name its record, what it is a member of."""
	record = member_record(scope, member)
	if record is not None:
		return Field(record=record.name, member=member.member)
	return _pointed(scope, member.base) if member.arrow else written(scope, member.base)


def _memory(scope: Scope, access: _Access) -> WriteTarget:
	"""A ``MEM`` store: of the object its address names, else of its pointee's record at the offset.

	An access no record holds writes every member of its own type, when that is a record.
	"""
	if access.base.startswith("&"):
		return written(scope, access.base[1:])
	width = _width(scope, access.type)
	pointed = pointed_member(scope, access.base, access.offset, width)
	cast = type_record(scope, access.cast)
	accessed = type_record(scope, access.type)
	if pointed is not None:
		return pointed
	if cast is not None:
		return _at_offset(scope, cast, _Span(offset=_signed(access.offset), width=width))
	return EveryMember(record=accessed.name) if accessed is not None else Unnamed()


def pointed_member(
	scope: Scope, pointer: str, offset: int, width: int | None
) -> WriteTarget | None:
	"""What the bytes at an offset from a pointer lie in, by the record it points to.

	None when it points to no record. An offset from a pointer to an array of records lands in one
	of their elements, in that array or a later row of the array holding it.
	"""
	declared = declared_type(scope, pointer)
	array = _ARRAY_POINTEE.match(declared.strip()) if declared is not None else None
	record = _pointee(scope, pointer)
	layout = record_layout(scope, record)
	signed = _signed(offset)
	if record is None:
		return None
	return _at_offset(
		scope,
		record,
		_Span(
			offset=(
				signed % layout.size
				if array is not None and layout is not None and layout.size > 0 and signed >= 0
				else signed
			),
			width=width,
		),
	)


def _signed(offset: int) -> int:
	"""A ``MEM`` or pointer offset, which GCC prints as its 32-bit unsigned wrap when negative."""
	return offset - _WRAP if offset >= _WRAP // 2 else offset


def _at_offset(scope: Scope, record: Record, span: _Span) -> WriteTarget:
	"""The innermost named member the bytes lie in, else every member; anything outside the record.

	A pointer offset before or past its record, as ``CONTAINER_OF`` makes, names no member of it.
	"""
	layout = record_layout(scope, record)
	if layout is None or layout.size == 0:
		return EveryMember(record=record.name)
	flexible = tuple(
		name
		for name, extent in layout.extents.items()
		if extent.size is None and layout.size <= extent.offset <= span.offset
	)
	if flexible:
		return Field(record=record.name, member=flexible[0])
	if not 0 <= span.offset < layout.size:
		return Unknown()
	match tuple(
		(name, extent)
		for name, extent in layout.extents.items()
		if extent.size is not None
		and span.width is not None
		and extent.offset <= span.offset
		and span.offset + span.width <= extent.offset + extent.size
	):
		case ((name, extent),) if extent.layout is not None:
			inner = _at_offset(
				scope,
				key_record(scope, extent.layout),
				_Span(offset=span.offset - extent.offset, width=span.width),
			)
			return (
				Field(record=record.name, member=name) if isinstance(inner, EveryMember) else inner
			)
		case ((name, _),):
			return Field(record=record.name, member=name)
		case _:
			return EveryMember(record=record.name)


def _width(scope: Scope, type_text: str) -> int | None:
	"""The bytes a type takes on 32-bit Arm, where the dump's names tell.

	A typedef of a scalar is a name the unit's types don't describe.
	"""
	text = _QUALIFIER.sub("", type_text).strip()
	array = _ARRAY_TYPE.match(text)
	layout = record_layout(scope, type_record(scope, text))
	if text.endswith("*"):
		return _POINTER_WIDTH
	if array is not None:
		element = _width(scope, array.group(1))
		return element * int(array.group(2)) if element is not None else None
	if layout is not None:
		return layout.size
	return _SCALAR_WIDTHS.get(text)


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
