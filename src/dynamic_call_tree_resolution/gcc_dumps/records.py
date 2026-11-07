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
from dynamic_call_tree_resolution.gcc_dumps.expressions import (
	ASSIGNMENT,
	CALL,
	NAMED_SSA,
	Scope,
	array_name,
	declared_type,
	function_scope,
	is_ssa,
	is_variable,
	member_record,
	record_layout,
	split_member,
	variable_symbol,
)
from dynamic_call_tree_resolution.model import FUNCTION_POINTER, Field, FunctionPointerMember

if TYPE_CHECKING:
	from pathlib import Path

	from dynamic_call_tree_resolution.descriptors import Descriptor
	from dynamic_call_tree_resolution.gcc_dumps.dump import DumpFunction, Statement
	from dynamic_call_tree_resolution.gcc_dumps.units import UnitTypes
	from dynamic_call_tree_resolution.model import StructureLayout

_OPERATION: Final = re.compile(r" [-+*/%&|^<>=!]+ |^\(.+\) ")
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


def unit_records(
	unit: str, functions: tuple[DumpFunction, ...], types: UnitTypes, root: Path
) -> tuple[DumpRecord, ...]:
	return tuple(
		record
		for function in functions
		for scope in (function_scope(types, function),)
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


def _located(unit: str, function: DumpFunction, statement: Statement, root: Path) -> Located:
	path = os.path.normpath(statement.file)
	return Located(
		unit=unit,
		function=function.symbol,
		file=os.path.relpath(path, root) if PurePath(path).is_relative_to(root) else path,
		line=statement.line,
		column=statement.column,
	)


def _records(scope: Scope, at: Located, body: str) -> tuple[DumpRecord, ...]:
	call = CALL.match(body)
	if call is not None:
		return (
			(SiteRecord(at=at, callee=_describe(scope, call.group(1))),)
			if _indirect(scope, call.group(1))
			else ()
		)
	assignment = ASSIGNMENT.match(body)
	return _store(scope, at, assignment.group(1), assignment.group(2)) if assignment else ()


def _indirect(scope: Scope, callee: str) -> bool:
	"""Whether a call goes through a value: a name of the unit's functions is called directly."""
	return is_ssa(scope, callee) and callee not in scope.types.functions


def _store(scope: Scope, at: Located, target: str, value: str) -> tuple[DumpRecord, ...]:
	"""The record of an assignment that stores a function pointer: a clobber stores nothing."""
	if value.startswith("{CLOBBER"):
		return ()
	member = split_member(target)
	if member is not None:
		record = member_record(scope, member)
		layout = record_layout(scope, record)
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
		(
			StoreRecord(
				at=at,
				place=Variable(symbol=variable_symbol(scope, target)),
				value=_describe(scope, value),
			),
		)
		if is_variable(scope, target) and _function_pointer_type(declared_type(scope, target))
		else ()
	)


def _describe(scope: Scope, value: str) -> Descriptor:
	named = NAMED_SSA.match(value)
	source = scope.definitions.get(value)
	if value in ("0B", "0"):
		return Null()
	if named is not None and named.group(2) and named.group(1) in scope.function.parameters:
		return Parameter(name=named.group(1))
	if value in scope.types.functions and declared_type(scope, value) is None:
		return Function(symbol=value)
	if source is not None and _OPERATION.search(source) is None:
		return _loaded(scope, value, source)
	return _other(scope, value)


def _other(scope: Scope, value: str) -> Other:
	r"""A value by its type, printed as the plugin prints it: without GCC's internal uids.

	>>> _UID.sub(r"<\1xxxx>", "int (*<T2cf>) (int)")
	'int (*<Txxxx>) (int)'
	"""
	return Other(type=_UID.sub(r"<\1xxxx>", declared_type(scope, value) or "<unknown>"))


def _loaded(scope: Scope, value: str, source: str) -> Descriptor:
	"""What a single load reads: a field, a variable or an array element."""
	member = split_member(source)
	array = array_name(source)
	if member is not None:
		record = member_record(scope, member)
		return (
			Field(record=record.name, member=member.member)
			if record is not None
			else _other(scope, value)
		)
	if is_variable(scope, source):
		return Variable(symbol=variable_symbol(scope, source))
	if array is not None:
		return Element(array=array)
	return _other(scope, value)


def _function_pointer_member(layout: StructureLayout, member: str) -> bool:
	return any(
		isinstance(candidate, FunctionPointerMember) and candidate.name == member
		for candidate in layout.members
	)


def _function_pointer_type(type_text: str | None) -> bool:
	return type_text is not None and ("(*" in type_text or type_text == FUNCTION_POINTER)


def _code(scope: Scope, value: str) -> bool:
	"""Whether a stored value is a function or a function pointer."""
	return value in scope.types.functions or _function_pointer_type(declared_type(scope, value))
