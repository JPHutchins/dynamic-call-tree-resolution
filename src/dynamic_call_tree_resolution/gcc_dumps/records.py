# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The descriptors plugin's records, rebuilt from a dump's functions and its unit's types."""

from __future__ import annotations

import os
import re
from pathlib import PurePath
from typing import TYPE_CHECKING, Final, assert_never

from salix import Struct

from dynamic_call_tree_resolution.descriptors import (
	Element,
	Function,
	Null,
	Other,
	Parameter,
	Variable,
	descriptor_fields,
)
from dynamic_call_tree_resolution.gcc_dumps.units import Record, anonymous_name
from dynamic_call_tree_resolution.model import (
	FUNCTION_POINTER,
	EmbeddedStructMember,
	Field,
	FunctionPointerMember,
)

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path

	from dynamic_call_tree_resolution.descriptors import Descriptor
	from dynamic_call_tree_resolution.gcc_dumps.dump import DumpFunction, Statement
	from dynamic_call_tree_resolution.gcc_dumps.units import UnitTypes
	from dynamic_call_tree_resolution.model import StructureLayout

_TEMPORARY: Final = re.compile(r"^_\d+$")
_NAMED_SSA: Final = re.compile(r"^([A-Za-z_][\w.$]*)_\d+(\(D\))?$")
_IDENTIFIER: Final = re.compile(r"^[A-Za-z_][\w.$]*$")
_CALL: Final = re.compile(r"^(?:\S+ = )?(\S+) \((.*)\)$")
_ASSIGNMENT: Final = re.compile(r"^(.+?) = (.+)$")
_MEMORY: Final = re.compile(r"^MEM(?: <.+?>)? ?\[\((.+?) \*\)(.+)\]$")
_ARRAY_MEMORY: Final = re.compile(r"^MEM <(.+?)\[.*?\]> ?\[\(")
_ELEMENT: Final = re.compile(r"^(.+)\[[^\[\]]+\]$")
_ANONYMOUS_MEMBER: Final = re.compile(r"^(.+?)(?:(->)|\.)(D\.\d+)$")
_OPERATION: Final = re.compile(r" [-+*/%&|^<>=!]+ |^\(.+\) ")
_QUALIFIER: Final = re.compile(r"\b(?:const|volatile)\b ?")
_UID: Final = re.compile(r"<([TD])[0-9a-f]+>")


class Located(Struct):
	"""Where a record's statement is: its unit and function, and the file, line and column."""

	unit: str
	function: str
	file: str
	"""Relative to the root, as the plugin's replay writes it."""
	line: int
	column: int


class SiteRecord(Struct):
	"""An indirect call and what it loads its callee from."""

	at: Located
	callee: Descriptor


class StoreRecord(Struct):
	"""A store of a function pointer into a field or a variable."""

	at: Located
	place: Field | Variable
	value: Descriptor


class Gap(Struct):
	"""A statement the dump shows but the unit's types can't place."""

	at: Located
	statement: str


type DumpRecord = SiteRecord | StoreRecord | Gap


class _Member(Struct):
	"""A member access, split into what it is taken from and the member."""

	base: str
	member: str
	arrow: bool


class _Scope(Struct):
	"""What describing one function's statements reads."""

	types: UnitTypes
	function: DumpFunction
	definitions: Mapping[str, str]
	"""Each name's defining assignment, where it is not a call."""


def unit_records(
	unit: str, functions: tuple[DumpFunction, ...], types: UnitTypes, root: Path
) -> tuple[DumpRecord, ...]:
	return tuple(
		record
		for function in functions
		for scope in (_scope(types, function),)
		for statement in function.statements
		for record in _records(scope, _located(unit, function, statement, root), statement.body)
	)


def record_line(record: SiteRecord | StoreRecord) -> str:
	"""The record as the plugin's replay writes it."""
	match record:
		case SiteRecord(at=at, callee=callee):
			return "\t".join(("site", *_location_fields(at), *descriptor_fields(callee)))
		case StoreRecord(at=at, place=place, value=value):
			return "\t".join(
				(
					"store",
					*_location_fields(at),
					*descriptor_fields(place),
					*descriptor_fields(value),
				)
			)
		case _ as unreachable:
			assert_never(unreachable)


def _location_fields(at: Located) -> tuple[str, ...]:
	return (at.unit, at.function, at.file, str(at.line), str(at.column))


def _scope(types: UnitTypes, function: DumpFunction) -> _Scope:
	return _Scope(
		types=types,
		function=function,
		definitions={
			match.group(1): match.group(2)
			for statement in function.statements
			if _CALL.match(statement.body) is None
			for match in (_ASSIGNMENT.match(statement.body),)
			if match is not None
		},
	)


def _located(unit: str, function: DumpFunction, statement: Statement, root: Path) -> Located:
	path = os.path.normpath(statement.file)
	return Located(
		unit=unit,
		function=function.symbol,
		file=os.path.relpath(path, root) if PurePath(path).is_relative_to(root) else path,
		line=statement.line,
		column=statement.column,
	)


def _records(scope: _Scope, at: Located, body: str) -> tuple[DumpRecord, ...]:
	call = _CALL.match(body)
	if call is not None:
		return (
			(SiteRecord(at=at, callee=_describe(scope, call.group(1))),)
			if _indirect(scope, call.group(1))
			else ()
		)
	assignment = _ASSIGNMENT.match(body)
	return _store(scope, at, assignment.group(1), assignment.group(2)) if assignment else ()


def _indirect(scope: _Scope, callee: str) -> bool:
	"""Whether a call goes through a value: a temporary, or an SSA version of a local or parameter."""
	named = _NAMED_SSA.match(callee)
	locals_ = scope.function.declarations.keys() | scope.function.parameters.keys()
	return _TEMPORARY.match(callee) is not None or (
		named is not None and (callee in locals_ or named.group(1) in locals_)
	)


def _store(scope: _Scope, at: Located, target: str, value: str) -> tuple[DumpRecord, ...]:
	member = _split_member(target)
	if member is not None:
		record = _member_record(scope, member)
		layout = _layout(scope, record)
		if record is None or layout is None:
			return (Gap(at=at, statement=f"{target} = {value}"),) if _code(scope, value) else ()
		return (
			(
				StoreRecord(
					at=at,
					place=Field(record=record.name, member=member.member),
					value=_describe(scope, value),
				),
			)
			if _function_pointer_member(layout, member.member)
			else ()
		)
	return (
		(StoreRecord(at=at, place=Variable(symbol=target), value=_describe(scope, value)),)
		if _variable(target) and _function_pointer_type(_declared(scope, target))
		else ()
	)


def _describe(scope: _Scope, value: str) -> Descriptor:
	named = _NAMED_SSA.match(value)
	source = scope.definitions.get(value)
	if value in ("0B", "0"):
		return Null()
	if named is not None and named.group(2) and named.group(1) in scope.function.parameters:
		return Parameter(name=named.group(1))
	if value in scope.types.functions and _declared(scope, value) is None:
		return Function(symbol=value)
	if source is not None and _OPERATION.search(source) is None:
		return _loaded(scope, value, source)
	return _other(scope, value)


def _other(scope: _Scope, value: str) -> Other:
	r"""A value by its type, printed as the plugin prints it: without GCC's internal uids.

	>>> _UID.sub(r"<\1xxxx>", "int (*<T2cf>) (int)")
	'int (*<Txxxx>) (int)'
	"""
	return Other(type=_UID.sub(r"<\1xxxx>", _declared(scope, value) or "<unknown>"))


def _loaded(scope: _Scope, value: str, source: str) -> Descriptor:
	"""What a single load reads: a field, a variable or an array element."""
	member = _split_member(source)
	element = _ELEMENT.match(source)
	if member is not None:
		record = _member_record(scope, member)
		return (
			Field(record=record.name, member=member.member)
			if record is not None
			else _other(scope, value)
		)
	if _variable(source):
		return Variable(symbol=source)
	if element is not None and _IDENTIFIER.match(element.group(1)):
		return Element(array=element.group(1))
	return _other(scope, value)


def _split_member(expression: str) -> _Member | None:
	"""The member an expression accesses last, outside every bracket and parenthesis.

	>>> _split_member("MEM[(const struct api *)_12].fetch")
	_Member(base='MEM[(const struct api *)_12]', member='fetch', arrow=False)
	>>> _split_member("timer_2(D)->expiry_fn"), _split_member("_5")
	(_Member(base='timer_2(D)', member='expiry_fn', arrow=True), None)
	"""
	match = re.match(r"^(.*?)(?:(->)|\.)([A-Za-z_]\w*)$", expression)
	return (
		_Member(base=match.group(1), member=match.group(3), arrow=match.group(2) is not None)
		if match is not None and _balanced(match.group(1))
		else None
	)


def _balanced(text: str) -> bool:
	return text.count("(") == text.count(")") and text.count("[") == text.count("]")


def _member_record(scope: _Scope, member: _Member) -> Record | None:
	"""The record a member is accessed in."""
	anonymous = _ANONYMOUS_MEMBER.match(member.base)
	if anonymous is not None:
		outer = _base_record(scope, anonymous.group(1), anonymous.group(2) is not None)
		return _anonymous(scope, outer, member.member)
	return _base_record(scope, member.base, member.arrow)


def _base_record(scope: _Scope, base: str, arrow: bool) -> Record | None:
	"""The record of what a member is taken from."""
	memory = _MEMORY.match(base)
	element = _ELEMENT.match(base)
	nested = _split_member(base)
	if arrow or (memory is None and element is None and nested is None):
		return _type_record(scope, _declared(scope, base))
	if memory is not None:
		return _type_record(scope, memory.group(1))
	if element is not None:
		array_memory = _ARRAY_MEMORY.match(element.group(1))
		return _type_record(
			scope,
			array_memory.group(1)
			if array_memory is not None
			else _declared(scope, element.group(1)),
		)
	return _nested_record(scope, nested)


def _nested_record(scope: _Scope, nested: _Member | None) -> Record | None:
	outer = _member_record(scope, nested) if nested is not None else None
	layout = _layout(scope, outer)
	extent = (
		layout.extents.get(nested.member) if layout is not None and nested is not None else None
	)
	key = extent.layout if extent is not None else None
	return _key_record(scope, key) if key is not None else None


def _anonymous(scope: _Scope, outer: Record | None, member: str) -> Record | None:
	"""The anonymous structure or union a member belongs to, inside the outer record."""
	layout = _layout(scope, outer)
	inside = tuple(
		embedded.type_name
		for embedded in (layout.members if layout is not None else ())
		if isinstance(embedded, EmbeddedStructMember)
		and embedded.name is None
		and _has_member(scope, embedded.type_name, member)
	)
	anywhere = tuple(
		key
		for key in scope.types.layouts
		if anonymous_name(key) != key and _has_member(scope, key, member)
	)
	match (inside or anywhere)[:2]:
		case (key,):
			return Record(name=anonymous_name(key), key=key)
		case _:
			return None


def _has_member(scope: _Scope, key: str, member: str) -> bool:
	layout = scope.types.layouts.get(key)
	return layout is not None and member in layout.extents


def _type_record(scope: _Scope, type_text: str | None) -> Record | None:
	"""The record a type names, through a pointer, an array or a typedef."""
	if type_text is None:
		return None
	text = re.sub(r"\s*\[[^\]]*\]$", "", _QUALIFIER.sub("", type_text).strip()).rstrip(" *")
	bare = text.split(" ", 1)[1] if text.startswith(("struct ", "union ")) else text
	return scope.types.typedefs.get(bare) or (
		Record(name=text, key=text) if text.startswith(("struct ", "union ")) else None
	)


def _key_record(scope: _Scope, key: str) -> Record:
	"""A layout key as a record: a typedef's key names the typedef's record."""
	return scope.types.typedefs.get(key) or Record(name=anonymous_name(key), key=key)


def _layout(scope: _Scope, record: Record | None) -> StructureLayout | None:
	return scope.types.layouts.get(record.key) if record is not None else None


def _function_pointer_member(layout: StructureLayout, member: str) -> bool:
	return any(
		isinstance(candidate, FunctionPointerMember) and candidate.name == member
		for candidate in layout.members
	)


def _declared(scope: _Scope, token: str) -> str | None:
	"""A name's type: as a local or SSA name, as a parameter, or as a file-scope variable."""
	named = _NAMED_SSA.match(token)
	base = named.group(1) if named is not None else token
	declarations = scope.function.declarations
	return (
		declarations.get(token)
		or declarations.get(base)
		or scope.function.parameters.get(base)
		or scope.types.variables.get(base)
	)


def _variable(text: str) -> bool:
	return (
		_IDENTIFIER.match(text) is not None
		and _NAMED_SSA.match(text) is None
		and _TEMPORARY.match(text) is None
	)


def _function_pointer_type(type_text: str | None) -> bool:
	return type_text is not None and ("(*" in type_text or type_text == FUNCTION_POINTER)


def _code(scope: _Scope, value: str) -> bool:
	"""Whether a stored value is a function or a function pointer."""
	return value in scope.types.functions or _function_pointer_type(_declared(scope, value))
