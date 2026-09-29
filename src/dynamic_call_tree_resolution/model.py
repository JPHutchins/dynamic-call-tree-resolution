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
	"""Return and parameter types of a function, as resolved type names."""

	return_type: str
	parameters: tuple[str, ...]


class Function(Struct):
	"""A named code region with an optional DWARF-derived signature."""

	name: str
	address: Address
	size: int
	signature: FunctionSignature | None


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
	"""A structure member pointing at a statically allocated struct instance.

	``pointee=None`` marks opaque (``void *``) pointers, which are followed
	when their static value names an object of any known structure type.
	"""

	kind: Literal["struct_pointer"]
	name: str | None
	offset: int
	pointee: str | None


class EmbeddedStructMember(Struct):
	"""A by-value structure member whose own members resolve in place."""

	kind: Literal["embedded_struct"]
	name: str | None
	offset: int
	members: tuple[Member, ...]


type Member = FunctionPointerMember | StructPointerMember | EmbeddedStructMember


class StructureLayout(Struct):
	"""Byte offsets of a structure's pointer-valued members, and its size."""

	members: tuple[Member, ...]
	size: int


class Relocation(Struct):
	"""A link-time fixup: a slot address pointing at a target address."""

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
	"""The instruction-aligned form of an address; identity when already even."""
	return Address(address & ~1)


def thumb_twin(address: Address) -> Address:
	"""The odd Thumb-bit form of an address, for symbol-table twins."""
	return Address(address | 1)


ANONYMOUS: Final = "<anonymous>"

FUNCTION_POINTER: Final = "function pointer"
ARRAY_SUFFIX: Final = " []"


def array_element_type(type_name: str) -> str:
	"""The element type name of an array type name; identity when not an array."""
	return type_name.removesuffix(ARRAY_SUFFIX)


def layout_key(keyword: str, name: str) -> str:
	"""The layout-table key for a structure or union type name."""
	return f"{keyword} {name}"


class Provenance(Enum):
	"""How a slot assignment was established."""

	RELOCATION = auto()
	CONSTANT_DATA = auto()


class SlotAssignment(Struct):
	"""Candidate target functions for one function-pointer slot."""

	slot: Address
	path: tuple[str | None, ...]
	candidates: frozenset[Address]
	provenance: Provenance


class CallSite(Struct):
	"""An indirect call or tail-branch instruction in one function's code.

	``slot`` is the address the call target is taken from when the
	operand's value set is a single address — the target function itself
	or the slot holding it — and ``None`` when the set could not be
	narrowed to one. ``candidates`` is the pre-chase set of addresses the
	value-set analysis tracked into the operand; an empty set means
	unresolved and consumers fall back to every address-taken function.
	"""

	caller_address: Address
	site_address: Address
	slot: Address | None
	candidates: frozenset[Address]


class UnresolvedSlot(Struct):
	"""A function-pointer slot with no statically resolved candidates."""

	slot: Address
	path: tuple[str | None, ...]
	signature: FunctionSignature | None


def render_path(path: tuple[str | None, ...]) -> str:
	"""Render a member path, naming anonymous members ``<anonymous>``.

	>>> render_path(("dev_a", "api", "open"))
	'dev_a.api.open'
	>>> render_path(("dev_a", None, "open"))
	'dev_a.<anonymous>.open'
	"""
	return ".".join("<anonymous>" if name is None else name for name in path)
