# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Immutable domain model for ELF programs and indirect-call resolution results."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Final, Literal, NewType

from salix import Struct, replace

if TYPE_CHECKING:
	from collections.abc import Mapping

Address = NewType("Address", int)

type ByteOrder = Literal["little", "big"]


class Machine(StrEnum):
	"""The ``e_machine`` values the analyzer can lift."""

	EM_X86_64 = "EM_X86_64"
	EM_386 = "EM_386"
	EM_ARM = "EM_ARM"

	@property
	def is_x86(self) -> bool:
		return self is Machine.EM_X86_64 or self is Machine.EM_386


class InstructionSet(StrEnum):
	"""The encodings the analyzer decodes."""

	A32 = "A32"
	T32 = "T32"
	X86_32 = "X86_32"
	X86_64 = "X86_64"


class FunctionSignature(Struct):
	"""A function's type, as resolved type names."""

	return_type: str
	parameters: tuple[str, ...]


class Function(Struct):
	"""A code region of the image."""

	name: str
	address: Address
	size: int
	signature: FunctionSignature | None
	"""From DWARF."""


class DataObject(Struct):
	"""A statically allocated data symbol."""

	name: str
	address: Address
	size: int
	type_name: str | None
	signature: FunctionSignature | None


class FunctionPointerMember(Struct):
	"""A structure member whose value is a function pointer."""

	kind: Literal["function_pointer"]
	name: str | None
	offset: int
	signature: FunctionSignature | None


class StructPointerMember(Struct):
	"""A structure member pointing at a statically allocated struct instance."""

	kind: Literal["struct_pointer"]
	name: str | None
	offset: int
	pointee: str | None
	"""``None`` for an opaque ``void *``."""


class EmbeddedStructMember(Struct):
	"""A by-value structure member."""

	kind: Literal["embedded_struct"]
	name: str | None
	offset: int
	members: tuple[Member, ...]
	"""Resolved in place."""


class ArrayMember(Struct):
	"""A by-value array member whose elements hold pointers."""

	kind: Literal["array"]
	name: str | None
	offset: int
	count: int
	stride: int
	element: Member
	"""One element, at offset 0."""


type Member = FunctionPointerMember | StructPointerMember | EmbeddedStructMember | ArrayMember


def array_elements(member: ArrayMember) -> tuple[Member, ...]:
	"""Each element as a member named by its index, at its offset within the array.

	>>> element = StructPointerMember(kind="struct_pointer", name=None, offset=0, pointee=None)
	>>> [
	...     (item.name, item.offset)
	...     for item in array_elements(
	...         ArrayMember(kind="array", name="slots", offset=8, count=2, stride=4, element=element)
	...     )
	... ]
	[('[0]', 0), ('[1]', 4)]
	"""
	return tuple(
		replace(member.element, name=f"[{index}]", offset=index * member.stride)
		for index in range(member.count)
	)


class StructureLayout(Struct):
	"""One structure type's layout."""

	members: tuple[Member, ...]
	"""Pointer-valued ones only."""
	size: int


class Relocation(Struct):
	"""A link-time fixup."""

	slot: Address
	target: Address
	addend: int
	type_name: str


class Section(Struct):
	"""One allocated section of the image, as linked."""

	data: bytes
	writable: bool


class Program(Struct):
	"""The immutable analysis model of one ELF image."""

	byte_order: ByteOrder
	pointer_size: int
	machine: Machine
	functions: Mapping[Address, Function]
	objects: Mapping[Address, DataObject]
	layouts: Mapping[str, StructureLayout]
	relocations: tuple[Relocation, ...]
	sections: Mapping[Address, Section]
	data_in_code: tuple[tuple[Address, Address], ...]
	"""Spans of data inside code, such as literal pools, from ARM `$d` mapping symbols."""
	arm_code: tuple[tuple[Address, Address], ...]
	"""Sorted spans of A32 code, from ARM `$a` mapping symbols."""


def aligned(address: Address) -> Address:
	return Address(address & ~1)


def thumb_twin(address: Address) -> Address:
	return Address(address | 1)


ANONYMOUS: Final = "<anonymous>"

FUNCTION_POINTER: Final = "function pointer"
ARRAY_SUFFIX: Final = " []"


def array_element_type(type_name: str) -> str:
	return type_name.removesuffix(ARRAY_SUFFIX)


def layout_key(keyword: str, name: str) -> str:
	return f"{keyword} {name}"


class Provenance(StrEnum):
	"""Whether a slot's image value is the only value it holds at runtime."""

	ROM_CONSTANT = "rom_constant"
	RAM_INITIALIZER = "ram_initializer"


class SlotAssignment(Struct):
	"""The static resolution of one stored function pointer."""

	slot: Address
	path: tuple[str | None, ...]
	candidates: frozenset[Address]
	provenance: Provenance
	relocated: bool
	"""A dynamic relocation supplies the value at load time."""


class CallSite(Struct):
	"""An indirect call or tail-branch instruction in one function's code."""

	caller_address: Address
	site_address: Address
	slot: Address | None
	"""Where the target is taken from, when the operand's value set is one address.

	For a register operand this is the target itself; for an x86 memory operand
	it is the address the target is loaded from.
	"""
	candidates: frozenset[Address]
	"""What the value-set analysis tracked into the operand, before chasing; empty when unresolved."""


class UnresolvedSlot(Struct):
	"""A stored function pointer the analysis cannot resolve."""

	slot: Address
	path: tuple[str | None, ...]
	signature: FunctionSignature | None


def render_path(path: tuple[str | None, ...]) -> str:
	"""Dotted form, with anonymous members named ``<anonymous>``.

	>>> render_path(("dev_a", "api", "open"))
	'dev_a.api.open'
	>>> render_path(("dev_a", None, "open"))
	'dev_a.<anonymous>.open'
	"""
	return ".".join("<anonymous>" if name is None else name for name in path)
