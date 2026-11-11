# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""What a dump's expressions refer to: records, members, arrays and variables, by the unit's types."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from salix import Struct

from dynamic_call_tree_resolution.composition import merged
from dynamic_call_tree_resolution.gcc_dumps.units import Record, anonymous_name
from dynamic_call_tree_resolution.model import ArrayMember, EmbeddedStructMember

if TYPE_CHECKING:
	from collections.abc import Mapping

	from dynamic_call_tree_resolution.gcc_dumps.dump import DumpFunction, Statement
	from dynamic_call_tree_resolution.gcc_dumps.units import UnitTypes
	from dynamic_call_tree_resolution.model import StructureLayout

_TEMPORARY: Final = re.compile(r"^_\d+$")
NAMED_SSA: Final = re.compile(r"^([A-Za-z_][\w.$]*)_\d+(\(D\))?$")
_IDENTIFIER: Final = re.compile(r"^[A-Za-z_][\w.$]*$")
CALL: Final = re.compile(r"^(?:.+? = )?(\S+) \((.*)\)$")
ASSIGNMENT: Final = re.compile(r"^(.+?) = (.+)$")
_MEMORY: Final = re.compile(r"^MEM(?: <(.+?)>)? ?\[\((.+?) \*(?: \{ref-all\})?\)(.+)\]$")
_ARRAY_MEMORY: Final = re.compile(r"^MEM <(.+?)\[.*?\]> ?\[\(")
_ELEMENT: Final = re.compile(r"^(.+)\[[^\[\]]+\]$")
_MEMORY_PREFIX: Final = re.compile(r"^MEM(?: <.+>)? ?$")
_DEREFERENCE: Final = re.compile(r"^\(\*(.+)\)$")
_DECLARATOR_SUFFIX: Final = re.compile(r"(?:\s*(?:\*|\[[^\]]*\]))+$")
_ANONYMOUS_MEMBER: Final = re.compile(r"^(.+?)(?:(->)|\.)(D\.\d+)$")
_MEMBER: Final = re.compile(r"^(.*?)(?:(->)|\.)([A-Za-z_]\w*)$")
_QUALIFIER: Final = re.compile(r"\b(?:const|volatile|static)\b ?")


class Scope(Struct):
	"""What describing one function's statements reads."""

	types: UnitTypes
	function: DumpFunction
	definitions: Mapping[str, str]
	"""Each name's defining assignment, where it is not a call."""
	locals: frozenset[str]
	"""The names the function declares, its parameters among them."""


class Member(Struct):
	"""A member access, split into what it is taken from and the member."""

	base: str
	member: str
	arrow: bool


def function_scope(types: UnitTypes, function: DumpFunction) -> Scope:
	return Scope(
		types=types,
		function=function,
		definitions=merged(map(_definition, function.statements)),
		locals=frozenset(function.declarations.keys() | function.parameters.keys()),
	)


def _definition(statement: Statement) -> dict[str, str]:
	assignment = ASSIGNMENT.match(statement.body)
	return (
		{assignment.group(1): assignment.group(2)}
		if assignment is not None and CALL.match(statement.body) is None
		else {}
	)


def member_record(scope: Scope, member: Member) -> Record | None:
	"""The record a member is accessed in."""
	anonymous = _ANONYMOUS_MEMBER.match(member.base)
	if anonymous is not None:
		outer = _base_record(scope, anonymous.group(1), anonymous.group(2) is not None)
		return _anonymous(scope, outer, member.member)
	return _base_record(scope, member.base, member.arrow)


def record_layout(scope: Scope, record: Record | None) -> StructureLayout | None:
	return scope.types.layouts.get(record.key) if record is not None else None


def declared_type(scope: Scope, token: str) -> str | None:
	"""A name's type: as a local or SSA name, as a parameter, or as a file-scope variable."""
	named = NAMED_SSA.match(token)
	base = named.group(1) if named is not None and is_ssa(scope, token) else token
	declarations = scope.function.declarations
	if token == "<retval>":
		return scope.function.result
	return (
		declarations.get(token)
		or declarations.get(base)
		or scope.function.parameters.get(base)
		or scope.types.variables.get(base)
	)


def type_record(scope: Scope, type_text: str | None) -> Record | None:
	"""The record a type names, through a pointer, an array or a typedef."""
	if type_text is None:
		return None
	text = _DECLARATOR_SUFFIX.sub("", _QUALIFIER.sub("", type_text).strip())
	tagged = text.startswith(("struct ", "union "))
	typedef = scope.types.typedefs.get(text.split(" ", 1)[1] if tagged else text)
	tag = Record(name=text, key=text) if tagged else None
	if typedef is None or tag is None:
		return typedef or tag
	return typedef if typedef.name == tag.name or tag.key not in scope.types.layouts else None


def is_variable(scope: Scope, text: str) -> bool:
	return (
		_IDENTIFIER.match(text) is not None
		and not is_ssa(scope, text)
		and text not in scope.function.parameters
	)


def variable_symbol(scope: Scope, variable: str) -> str:
	"""A variable's symbol: a function-scope static's is numbered, ``hook.0``."""
	declared = scope.function.declarations.get(variable)
	return (
		scope.types.statics.get(variable, variable)
		if declared is not None and declared.startswith("static ")
		else variable
	)


def is_ssa(scope: Scope, token: str) -> bool:
	"""Whether a name is an SSA name here: a temporary, or a version of a local or parameter.

	A file-scope ``api_0`` is no version of anything the function declares, and a static named
	``state_1`` is a variable.
	"""
	named = NAMED_SSA.match(token)
	declared = scope.function.declarations.get(token, "")
	return not declared.startswith("static ") and (
		_TEMPORARY.match(token) is not None
		or (named is not None and (token in scope.locals or named.group(1) in scope.locals))
	)


def split_member(expression: str) -> Member | None:
	"""The member an expression accesses last, outside every bracket and parenthesis.

	>>> split_member("MEM[(const struct api *)_12].fetch")
	Member(base='MEM[(const struct api *)_12]', member='fetch', arrow=False)
	>>> split_member("timer_2(D)->expiry_fn"), split_member("_5")
	(Member(base='timer_2(D)', member='expiry_fn', arrow=True), None)
	"""
	match = _MEMBER.match(expression)
	return (
		Member(base=match.group(1), member=match.group(3), arrow=match.group(2) is not None)
		if match is not None and _balanced(match.group(1))
		else None
	)


def array_name(expression: str) -> str | None:
	"""The array an element is read from, by its name: the brackets of a ``MEM`` hold an address.

	>>> array_name("table[i_2(D)]"), array_name("d_3(D)->tbl[i_4(D)]")
	('table', '<expression>')
	>>> array_name("MEM[(int *)p_1(D)]") is None
	True
	"""
	array = indexed_array(expression)
	if array is None:
		return None
	return array if _IDENTIFIER.match(array) else "<expression>"


def indexed_array(expression: str) -> str | None:
	"""What an element is indexed from, unless it is a ``MEM``, whose brackets hold an address.

	>>> indexed_array("d_3(D)->tbl[i_4(D)]"), indexed_array("MEM[(int *)p_1(D)]")
	('d_3(D)->tbl', None)
	"""
	element = _ELEMENT.match(expression)
	if element is None or _MEMORY_PREFIX.match(element.group(1)):
		return None
	return element.group(1)


def named_member(member: Member) -> Member:
	"""The member as its nearest named record holds it, through anonymous structs and unions.

	>>> named_member(Member(base="prev_25->D.4736", member="next", arrow=False))
	Member(base='prev_25', member='next', arrow=True)
	"""
	anonymous = _ANONYMOUS_MEMBER.match(member.base)
	return (
		named_member(
			Member(
				base=anonymous.group(1), member=member.member, arrow=anonymous.group(2) is not None
			)
		)
		if anonymous is not None
		else member
	)


def _balanced(text: str) -> bool:
	return text.count("(") == text.count(")") and text.count("[") == text.count("]")


def _base_record(scope: Scope, base: str, arrow: bool) -> Record | None:
	"""The record of what a member is taken from."""
	memory = _MEMORY.match(base)
	element = _ELEMENT.match(base)
	indexed = element is not None and _MEMORY_PREFIX.match(element.group(1)) is None
	nested = split_member(base)
	if arrow or (memory is None and element is None and nested is None):
		return type_record(scope, declared_type(scope, base))
	if element is not None and indexed:
		return _array_record(scope, element.group(1))
	if memory is not None:
		return type_record(scope, _accessed(memory.group(1), memory.group(2)))
	return _nested_record(scope, nested)


def _accessed(access: str | None, cast: str) -> str:
	"""The type a ``MEM`` accesses: its own, which a member is taken from, else its address's cast.

	>>> _accessed("struct z_heap", "struct z_heap_bucket"), _accessed(None, "struct k_timer")
	('struct z_heap', 'struct k_timer')
	"""
	return access if access is not None else cast


def _array_record(scope: Scope, array: str) -> Record | None:
	"""The record of an array's elements: of a ``MEM``, a member, an element, or a declared name."""
	array_memory = _ARRAY_MEMORY.match(array)
	member = split_member(array)
	element = _ELEMENT.match(array)
	pointed = _DEREFERENCE.match(array)
	if array_memory is not None:
		return type_record(scope, array_memory.group(1))
	if member is not None:
		return _element_record(scope, member)
	if element is not None:
		return _array_record(scope, element.group(1))
	return type_record(
		scope, declared_type(scope, pointed.group(1) if pointed is not None else array)
	)


def _element_record(scope: Scope, array: Member) -> Record | None:
	"""The record of an element of an array member."""
	layout = record_layout(scope, member_record(scope, array))
	match tuple(
		candidate.element.type_name
		for candidate in (layout.members if layout is not None else ())
		if isinstance(candidate, ArrayMember)
		and candidate.name == array.member
		and isinstance(candidate.element, EmbeddedStructMember)
	):
		case (key,):
			return key_record(scope, key)
		case _:
			return None


def _nested_record(scope: Scope, nested: Member | None) -> Record | None:
	outer = member_record(scope, nested) if nested is not None else None
	layout = record_layout(scope, outer)
	extent = (
		layout.extents.get(nested.member) if layout is not None and nested is not None else None
	)
	key = extent.layout if extent is not None else None
	return key_record(scope, key) if key is not None else None


def _anonymous(scope: Scope, outer: Record | None, member: str) -> Record | None:
	"""The anonymous structure or union a member belongs to, inside the outer record."""
	layout = record_layout(scope, outer)
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


def _has_member(scope: Scope, key: str, member: str) -> bool:
	layout = scope.types.layouts.get(key)
	return layout is not None and member in layout.extents


def key_record(scope: Scope, key: str) -> Record:
	"""A layout key as a record: a typedef's key names the typedef's record."""
	return scope.types.typedefs.get(key) or Record(name=anonymous_name(key), key=key)
