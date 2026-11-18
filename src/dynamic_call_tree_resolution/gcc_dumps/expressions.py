# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""What a dump's expressions refer to: records, members, arrays and variables, by the unit's types."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from salix import Struct

from dynamic_call_tree_resolution.composition import merged, unanimous
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
CV_QUALIFIER: Final = re.compile(r"\b(?:const|volatile)\b ?")
_ARITHMETIC: Final = re.compile(
	r"^(?:(?:signed|unsigned|char|short|int|long|_Bool|float|double)\b ?)+$"
)


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
		outers = _base_records(scope, anonymous.group(1), anonymous.group(2) is not None)
		return unanimous((_anonymous(scope, outer, member.member) for outer in outers), None)
	return unanimous(_base_records(scope, member.base, member.arrow), None)


def record_layout(scope: Scope, record: Record | None) -> StructureLayout | None:
	return scope.types.layouts.get(record.key) if record is not None else None


def possible_types(scope: Scope, token: str) -> frozenset[str | None]:
	"""The types a name may have: as a local or SSA name or a parameter, else a file-scope variable.

	Inlined scopes may declare a name again, and an SSA name may be a version of any of them that is
	no structure, union or array. Declarations that differ only in qualifiers give one type.
	"""
	if token == "<retval>":
		return frozenset((scope.function.result,))
	named = NAMED_SSA.match(token)
	ssa = named is not None and is_ssa(scope, token)
	base = named.group(1) if named is not None and ssa else token
	declared = _declared(scope, token, base)
	versioned = ssa and token not in scope.function.declarations
	candidates = tuple(
		candidate for candidate in declared if not (versioned and _aggregate(scope, candidate))
	)
	if not declared:
		return frozenset((scope.types.variables.get(base),))
	if len(frozenset(map(_unqualified, candidates))) == 1:
		return frozenset(candidates[-1:])
	return frozenset(candidates)


def pointer_types(scope: Scope, token: str) -> frozenset[str | None]:
	"""The types a name the dump uses as a pointer may have: all of them, where none may be one."""
	possible = possible_types(scope, token)
	pointers = frozenset(
		declared for declared in possible if declared is None or _may_point(scope, declared)
	)
	return pointers or possible


def _declared(scope: Scope, token: str, base: str) -> tuple[str, ...]:
	"""A name's declarations in the function: an SSA name's own, else its base's, a parameter's first."""
	parameter = scope.function.parameters.get(base)
	own = scope.function.declarations.get(token, ()) if token != base else ()
	return own or (
		*((parameter,) if parameter is not None else ()),
		*scope.function.declarations.get(base, ()),
	)


def _aggregate(scope: Scope, type_text: str) -> bool:
	"""Whether a type is a structure, union or array, which no SSA name has.

	GCC prints a pointer to an array as ``int[4] *``.
	"""
	text = _unqualified(type_text)
	return text.endswith("]") or (
		"*" not in text
		and (text.startswith(("struct ", "union ")) or type_record(scope, text) is not None)
	)


def _accessible(scope: Scope, type_text: str, arrow: bool) -> bool:
	"""Whether a member access may be of a name of a type: ``->`` needs a pointer, ``.`` a record."""
	return (
		_may_point(scope, type_text)
		if arrow
		else "*" not in type_text and _ARITHMETIC.match(_unqualified(type_text)) is None
	)


def _may_point(scope: Scope, type_text: str) -> bool:
	return not _aggregate(scope, type_text) and _ARITHMETIC.match(_unqualified(type_text)) is None


def _unqualified(type_text: str) -> str:
	return CV_QUALIFIER.sub("", type_text).strip()


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
	return scope.types.statics.get(variable, variable) if is_static(scope, variable) else variable


def is_ssa(scope: Scope, token: str) -> bool:
	"""Whether a name is an SSA name here: a temporary, or a version of a local or parameter.

	A file-scope ``api_0`` is no version of anything the function declares, and a static named
	``state_1`` is a variable.
	"""
	named = NAMED_SSA.match(token)
	return not is_static(scope, token) and (
		_TEMPORARY.match(token) is not None
		or (named is not None and (token in scope.locals or named.group(1) in scope.locals))
	)


def is_static(scope: Scope, name: str) -> bool:
	return any(
		declared.startswith("static ") for declared in scope.function.declarations.get(name, ())
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


def _base_records(scope: Scope, base: str, arrow: bool) -> frozenset[Record | None]:
	"""The record of what a member is taken from, for each type a name may have."""
	memory = _MEMORY.match(base)
	element = _ELEMENT.match(base)
	indexed = element is not None and _MEMORY_PREFIX.match(element.group(1)) is None
	nested = split_member(base)
	if arrow or (memory is None and element is None and nested is None):
		return frozenset(
			type_record(scope, declared)
			for declared in possible_types(scope, base)
			if declared is None or _accessible(scope, declared, arrow)
		)
	if element is not None and indexed:
		return frozenset((_array_record(scope, element.group(1)),))
	if memory is not None:
		return frozenset((type_record(scope, _accessed(memory.group(1), memory.group(2))),))
	return frozenset((_nested_record(scope, nested),))


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
	return unanimous(
		(
			type_record(scope, declared)
			for declared in possible_types(
				scope, pointed.group(1) if pointed is not None else array
			)
		),
		None,
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
