# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Immutable domain model for ELF programs and indirect-call resolution results."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Final, Literal, NewType, assert_never

from salix import Struct, replace

if TYPE_CHECKING:
	from collections.abc import Mapping

	from dynamic_call_tree_resolution.vsa.lattice import ValueSet

Address = NewType("Address", int)

type ByteOrder = Literal["little", "big"]


class InstructionFamily(StrEnum):
	"""The instruction architectures whose code the analyzer decodes and interprets."""

	ARM = "ARM"
	X86 = "X86"


class Machine(StrEnum):
	"""The ``e_machine`` values the analyzer can lift."""

	EM_X86_64 = "EM_X86_64"
	EM_386 = "EM_386"
	EM_ARM = "EM_ARM"

	@property
	def family(self) -> InstructionFamily:
		match self:
			case Machine.EM_X86_64 | Machine.EM_386:
				return InstructionFamily.X86
			case Machine.EM_ARM:
				return InstructionFamily.ARM
			case _ as unreachable:
				assert_never(unreachable)


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
	type_name: str
	"""Its layout's key in ``Program.layouts``."""
	members: tuple[Member, ...]
	"""Resolved in place."""


class SkipReason(StrEnum):
	"""Why something that may hold or lead to code has no enumerated slots."""

	POINTER_TO_POINTER = "pointer to a pointer"
	POINTER_TO_ARRAY = "pointer to an array"
	UNSIZED_ARRAY = "array of unknown size"
	UNTYPED_OBJECT = "object without a type"


class SkippedMember(Struct):
	"""A member whose slots the analysis does not enumerate."""

	kind: Literal["skipped"]
	name: str | None
	offset: int
	reason: SkipReason


class ArrayMember(Struct):
	"""A by-value array member whose elements hold pointers."""

	kind: Literal["array"]
	name: str | None
	offset: int
	count: int
	stride: int
	element: Member
	"""One element, at offset 0."""


type Member = (
	FunctionPointerMember | StructPointerMember | EmbeddedStructMember | ArrayMember | SkippedMember
)


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
	"""Only those that hold or may lead to code."""
	size: int


class Relocation(Struct):
	"""A link-time fixup."""

	slot: Address
	target: Address
	addend: int
	type_name: str


class ReferenceKind(StrEnum):
	"""What a reference the linker resolved does with its symbol."""

	ADDRESS = "address"
	CALL = "call"
	OTHER = "other"


class LinkReference(Struct):
	"""A reference the linker resolved and kept in the image (``--emit-relocs``)."""

	slot: Address
	symbol: str
	"""Empty for a reference to a whole section."""
	value: Address
	kind: ReferenceKind


class SourceLocation(Struct):
	"""A place in the source, by file name, line and column."""

	file: str
	line: int
	column: int


class LineSpan(Struct):
	"""Code that the line table places at one location."""

	start: Address
	end: Address
	location: SourceLocation


class Declaration(Struct):
	"""A function's declaration, within the compilation unit that defines it."""

	unit: str
	"""The unit's source, as DWARF names it."""
	location: SourceLocation


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
	link_references: tuple[LinkReference, ...] = ()
	"""The references into allocated sections that the linker kept; none without ``--emit-relocs``."""
	inlined: Mapping[str, tuple[tuple[Address, Address], ...]] = {}
	"""The code spans of each function's inlined copies, from DWARF."""
	declarations: Mapping[Address, Declaration] = {}
	"""Each function's declaration, by where its code starts; a clone has its origin's."""
	symbol_addresses: Mapping[str, frozenset[Address]] = {}
	"""Where each function symbol's code starts, aliases included."""
	arm_core: ArmCore | None = None
	"""The ARM core the build attributes target; ``None`` for other machines."""
	tls_size: int = 0
	"""The bytes of each thread's copy of thread-local storage (TLS), from the TLS segment."""
	entry_point: Address = Address(0)
	"""Where execution starts, from the ELF header."""


class ArmProfile(StrEnum):
	"""An ARM architecture profile, as ``Tag_CPU_arch_profile`` names it."""

	APPLICATION = "A"
	REAL_TIME = "R"
	MICROCONTROLLER = "M"


class ArmCore(Struct):
	"""What the ARM build attributes say about the core."""

	profile: ArmProfile
	floating_point: bool
	"""Whether the build targets an FP extension (``Tag_FP_arch``)."""


class ThreadRoot(Struct):
	"""A thread the image defines statically, as its RTOS starts it."""

	name: str
	entry: Address
	entry_slot: Address
	"""Where the thread's record holds the entry."""
	arguments: tuple[Address, ...]
	"""What the entry is started with, by ABI argument position."""


class SystemThread(Struct):
	"""A thread the RTOS creates for itself while it starts, through the same trampoline."""

	name: str
	entry: Address


class ThreadCreation(Struct):
	"""How an RTOS's thread creations reach its trampoline."""

	frame_builders: tuple[str, ...]
	"""The functions that build a thread's first frame, which enters the trampoline with the
	builder's function-pointer argument."""
	setup: str
	"""The function every thread creation passes the thread's entry to."""
	static_start: str
	"""The function that creates each static thread from its record."""
	static_entries: frozenset[Address] | None
	"""The entry of every static thread record; none when a record could not be read."""


class RtosModel(Struct):
	"""What an RTOS adds to the analysis of an image."""

	name: str
	evidence: tuple[str, ...]
	"""What identified the RTOS in the image."""
	threads: tuple[ThreadRoot, ...]
	trampoline: str | None
	"""The function that calls each thread's entry."""
	creation: ThreadCreation | None = None
	stack_reservation: int = 0
	"""The bytes the RTOS takes from the top of each thread's stack before the thread's entry runs."""
	exception_frame: int = 0
	"""The bytes an interrupt's hardware frame adds to the interrupted thread's stack."""
	system_threads: tuple[SystemThread, ...] = ()
	interrupt_stack: str | None = None
	"""The stack exception handlers run on, which an M-profile core's main stack pointer (MSP)
	addresses."""


BARE_METAL: Final = RtosModel(name="none", evidence=(), threads=(), trampoline=None)


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


class Unreached(Struct):
	"""A site in a block that the function's control flow never reaches."""


class CallSite(Struct):
	"""An indirect call or tail-branch instruction in one function's code."""

	caller_address: Address
	site_address: Address
	slot: Address | None
	"""Where the target is taken from, when the operand's value set is one address.

	For a register operand this is the target itself; for an x86 memory operand
	it is the address the target is loaded from.
	"""
	target: Unreached | ValueSet
	"""What the value-set analysis tracked into the operand, before chasing."""


class Residue(StrEnum):
	"""What an unresolved slot holds in the image, and whether that can change."""

	ROM_NULL = "rom_null"
	ROM_NON_FUNCTION = "rom_non_function"
	RAM_NULL = "ram_null"
	RAM_UNINITIALIZED = "ram_uninitialized"
	RAM_INITIALIZED = "ram_initialized"


class UnresolvedSlot(Struct):
	"""A stored function pointer the analysis cannot resolve."""

	slot: Address
	path: tuple[str | None, ...]
	signature: FunctionSignature | None
	residue: Residue


class NotEnumerated(Struct):
	"""An object or member that may hold or lead to code, left out of the slot universe."""

	path: tuple[str | None, ...]
	reason: SkipReason


def render_path(path: tuple[str | None, ...]) -> str:
	"""Dotted form, with anonymous members named ``<anonymous>``.

	>>> render_path(("dev_a", "api", "open"))
	'dev_a.api.open'
	>>> render_path(("dev_a", None, "open"))
	'dev_a.<anonymous>.open'
	"""
	return ".".join(ANONYMOUS if name is None else name for name in path)
