# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Immutable domain model for ELF programs and indirect-call resolution results."""

from __future__ import annotations

from enum import Enum, StrEnum, auto
from typing import TYPE_CHECKING, Final, Literal, NewType

from salix import Struct

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


type Member = FunctionPointerMember | StructPointerMember | EmbeddedStructMember


class StructureLayout(Struct):
	"""One structure type's layout."""

	members: tuple[Member, ...]
	"""Pointer-valued members only."""
	size: int


class Relocation(Struct):
	"""A link-time fixup."""

	slot: Address
	target: Address
	addend: int
	type_name: str


class Program(Struct):
	"""The immutable analysis model of one ELF image."""

	byte_order: ByteOrder
	pointer_size: int
	machine: Machine
	functions: Mapping[Address, Function]
	objects: Mapping[Address, DataObject]
	layouts: Mapping[str, StructureLayout]
	relocations: tuple[Relocation, ...]
	sections: Mapping[Address, bytes]


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


class Provenance(Enum):
	"""How a slot assignment was established."""

	RELOCATION = auto()
	CONSTANT_DATA = auto()


class SlotAssignment(Struct):
	"""The static resolution of one function-pointer slot."""

	slot: Address
	path: tuple[str | None, ...]
	candidates: frozenset[Address]
	provenance: Provenance


class CallSite(Struct):
	"""An indirect call or tail-branch instruction in one function's code."""

	caller_address: Address
	site_address: Address
	slot: Address | None
	"""Where the target is taken from, when the operand's value set is one address.

	For a register operand this is the target itself; for an x86 memory operand
	it is the slot holding the target.
	"""
	candidates: frozenset[Address]
	"""What the value-set analysis tracked into the operand, before chasing; empty when unresolved."""


class UnresolvedSlot(Struct):
	"""A function-pointer slot with no statically resolved candidates."""

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
