# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Immutable domain model for ELF programs and indirect-call resolution results."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Literal, NamedTuple, NewType

if TYPE_CHECKING:
	from collections.abc import Mapping

Address = NewType("Address", int)

type ByteOrder = Literal["little", "big"]


class FunctionSignature(NamedTuple):
	"""Return and parameter types of a function, as resolved type names."""

	return_type: str
	parameters: tuple[str, ...]


class Function(NamedTuple):
	"""A named code region with an optional DWARF-derived signature."""

	name: str
	address: Address
	size: int
	signature: FunctionSignature | None


class DataObject(NamedTuple):
	"""A statically allocated data symbol, with its initialized bytes."""

	name: str
	address: Address
	size: int
	type_name: str | None
	bytes: bytes


class FunctionPointerMember(NamedTuple):
	"""A structure member whose value is a function pointer."""

	kind: Literal["function_pointer"]
	name: str | None
	offset: int
	signature: FunctionSignature | None


class StructPointerMember(NamedTuple):
	"""A structure member pointing at a statically allocated struct instance."""

	kind: Literal["struct_pointer"]
	name: str | None
	offset: int
	pointee: str


type Member = FunctionPointerMember | StructPointerMember


class StructureLayout(NamedTuple):
	"""Byte offsets of a structure's pointer-valued members."""

	members: tuple[Member, ...]


class Relocation(NamedTuple):
	"""A link-time fixup: a slot address pointing at a target address."""

	slot: Address
	target: Address
	addend: int
	type_name: str


class Program(NamedTuple):
	"""The immutable analysis model of one ELF image."""

	byte_order: ByteOrder
	pointer_size: int
	functions: Mapping[Address, Function]
	objects: Mapping[Address, DataObject]
	layouts: Mapping[str, StructureLayout]
	relocations: tuple[Relocation, ...]


class Provenance(Enum):
	"""How a slot assignment was established."""

	RELOCATION = auto()
	CONSTANT_DATA = auto()


class SlotAssignment(NamedTuple):
	"""Candidate target functions for one function-pointer slot."""

	slot: Address
	path: tuple[str | None, ...]
	candidates: frozenset[Address]
	provenance: Provenance


def render_path(path: tuple[str | None, ...]) -> str:
	"""Render a member path, naming anonymous members ``<anonymous>``.

	>>> render_path(("dev_a", "api", "open"))
	'dev_a.api.open'
	>>> render_path(("dev_a", None, "open"))
	'dev_a.<anonymous>.open'
	"""
	return ".".join("<anonymous>" if name is None else name for name in path)
