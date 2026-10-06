# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Zephyr's static threads, from the ``K_THREAD_DEFINE`` records in the image."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, assert_never

from dynamic_call_tree_resolution.model import (
	Address,
	ArmCore,
	ArmProfile,
	Machine,
	RtosModel,
	ThreadCreation,
	ThreadRoot,
)
from dynamic_call_tree_resolution.points_to import in_writable_memory, pointer_at

if TYPE_CHECKING:
	from dynamic_call_tree_resolution.model import DataObject, Program, StructureLayout

_RECORD: Final = "struct _static_thread_data"
_TRAMPOLINE: Final = "z_thread_entry"
_RECORD_PREFIX: Final = "_k_thread_data_"
_ENTRY: Final = "init_entry"
_ARGUMENTS: Final = ("init_p1", "init_p2", "init_p3")
_FRAME_BUILDERS: Final = ("arch_new_thread", "arch_switch_to_main_thread")
_SETUP: Final = "z_setup_new_thread"
_STATIC_START: Final = "z_init_static_threads"
_BASIC_FRAME: Final = 8 * 4
_EXTENDED_FRAME: Final = 26 * 4
_ALIGNMENT_PAD: Final = 4
_TLS_SETUP: Final = "arch_tls_stack_setup"
_TOOLCHAIN_TLS_POINTERS: Final = 2
_STACK_POINTER_ALIGNMENT: Final = 8


def detect(program: Program) -> RtosModel | None:
	layout = program.layouts.get(_RECORD)
	if layout is None or all(
		function.name != _TRAMPOLINE for function in program.functions.values()
	):
		return None
	records = tuple(
		record
		for record in sorted(program.objects.values(), key=_address)
		if record.type_name == _RECORD
	)
	threads = tuple(
		thread for record in records if (thread := _thread(program, layout, record)) is not None
	)
	return RtosModel(
		name="zephyr",
		evidence=(_TRAMPOLINE, _RECORD),
		threads=threads,
		trampoline=_TRAMPOLINE,
		stack_reservation=_stack_reservation(program),
		exception_frame=_exception_frame(program.arm_core),
		creation=ThreadCreation(
			frame_builders=_FRAME_BUILDERS,
			setup=_SETUP,
			static_start=_STATIC_START,
			static_entries=(
				frozenset(thread.entry for thread in threads)
				if len(threads) == len(records)
				else None
			),
		),
	)


def _stack_reservation(program: Program) -> int:
	match program.machine:
		case Machine.EM_ARM:
			return (
				_tls_reservation(program.tls_size, program.pointer_size)
				if _TLS_SETUP in program.symbol_addresses
				else 0
			)
		case Machine.EM_386 | Machine.EM_X86_64:
			return 0
		case _ as unreachable:
			assert_never(unreachable)


def _tls_reservation(tls_size: int, pointer_size: int) -> int:
	"""What an ARM thread's stack gives to thread-local storage (TLS) before its entry runs.

	The TLS copy and two toolchain pointers sit at the top of the stack, and the stack
	pointer starts below them, aligned down to 8 bytes: ``ARCH_STACK_PTR_ALIGN`` with
	``CONFIG_STACK_ALIGN_DOUBLE_WORD``, which also covers the 4 without it.

	>>> _tls_reservation(4, 4)
	16
	>>> _tls_reservation(0, 4)
	8
	>>> _tls_reservation(12, 4)
	24
	"""
	return (
		(tls_size + _TOOLCHAIN_TLS_POINTERS * pointer_size + _STACK_POINTER_ALIGNMENT - 1)
		// _STACK_POINTER_ALIGNMENT
		* _STACK_POINTER_ALIGNMENT
	)


def _exception_frame(core: ArmCore | None) -> int:
	"""What an M-profile core stacks on the thread's stack when an interrupt takes it.

	ARMv7-M stacks a basic frame of 8 words, or 26 with a floating-point context, and may
	pad a word to align the stack to 8 bytes. Handlers then run on the main stack, so a
	nested interrupt adds nothing more. Other profiles are not modeled.

	>>> _exception_frame(ArmCore(profile=ArmProfile.MICROCONTROLLER, floating_point=False))
	36
	>>> _exception_frame(ArmCore(profile=ArmProfile.MICROCONTROLLER, floating_point=True))
	108
	>>> _exception_frame(ArmCore(profile=ArmProfile.APPLICATION, floating_point=True))
	0
	"""
	match core:
		case ArmCore(profile=ArmProfile.MICROCONTROLLER, floating_point=floating_point):
			return (_EXTENDED_FRAME if floating_point else _BASIC_FRAME) + _ALIGNMENT_PAD
		case ArmCore() | None:
			return 0
		case _ as unreachable:
			assert_never(unreachable)


def _address(data_object: DataObject) -> Address:
	return data_object.address


def _thread(program: Program, layout: StructureLayout, record: DataObject) -> ThreadRoot | None:
	offsets = {member.name: member.offset for member in layout.members}
	match (
		in_writable_memory(program, record.address),
		_word(program, record, offsets.get(_ENTRY)),
		tuple(_word(program, record, offsets.get(name)) for name in _ARGUMENTS),
	):
		case (False, int() as entry, (int() as p1, int() as p2, int() as p3)) if (
			entry in program.functions
		):
			return ThreadRoot(
				name=record.name.removeprefix(_RECORD_PREFIX),
				entry=Address(entry),
				entry_slot=Address(record.address + offsets[_ENTRY]),
				arguments=(Address(p1), Address(p2), Address(p3)),
			)
		case _:
			return None


def _word(program: Program, record: DataObject, offset: int | None) -> Address | None:
	return pointer_at(program, Address(record.address + offset)) if offset is not None else None
