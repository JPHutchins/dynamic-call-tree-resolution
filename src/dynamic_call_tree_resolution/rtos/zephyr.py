# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Zephyr's static threads, from the ``K_THREAD_DEFINE`` records in the image."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final, assert_never

from salix import Struct

from dynamic_call_tree_resolution.model import (
	Address,
	ArmCore,
	ArmProfile,
	InterruptStack,
	Machine,
	RtosModel,
	SystemThread,
	ThreadCreation,
	ThreadRoot,
)
from dynamic_call_tree_resolution.points_to import in_writable_memory, pointer_at

if TYPE_CHECKING:
	from collections.abc import Mapping

	from dynamic_call_tree_resolution.model import DataObject, Program, StructureLayout

_RECORD: Final = "struct _static_thread_data"
_TRAMPOLINE: Final = "z_thread_entry"
_RECORD_PREFIX: Final = "_k_thread_data_"
_ENTRY: Final = "init_entry"
_ARGUMENTS: Final = ("init_p1", "init_p2", "init_p3")
_STACK_SIZE: Final = "init_stack_size"
_FRAME_BUILDERS: Final = ("arch_new_thread", "arch_switch_to_main_thread")
_SETUP: Final = "z_setup_new_thread"
_STATIC_START: Final = "z_init_static_threads"
_BASIC_FRAME: Final = 8 * 4
_EXTENDED_FRAME: Final = 26 * 4
_ALIGNMENT_PAD: Final = 4
_TLS_SETUP: Final = "arch_tls_stack_setup"
_TOOLCHAIN_TLS_POINTERS: Final = 2
_DOUBLE_WORD: Final = 8
_SINGLE_WORD: Final = 4
_TLS: Final = "CONFIG_THREAD_LOCAL_STORAGE"
_RANDOM_OFFSET: Final = "CONFIG_STACK_POINTER_RANDOM"
_ALIGN_DOUBLE_WORD: Final = "CONFIG_STACK_ALIGN_DOUBLE_WORD"
_SYSTEM_THREADS: Final = (
	("z_main_thread", "bg_thread_main", "CONFIG_MAIN_STACK_SIZE"),
	("z_idle_threads", "idle", "CONFIG_IDLE_STACK_SIZE"),
)
_INTERRUPT_STACK: Final = "z_interrupt_stacks"
_INTERRUPT_STACK_SIZE: Final = "CONFIG_ISR_STACK_SIZE"
_KCONFIG_OPTION: Final = re.compile(r"^(CONFIG_\w+)=(.*)$", re.MULTILINE)
_KCONFIG_INTEGER: Final = re.compile(r"0x[0-9a-fA-F]+|[0-9]+")
_ARCH: Final = "CONFIG_ARCH"
_STACK_NOT_ALL_USABLE: Final = ("CONFIG_ARCH_POSIX", "CONFIG_MPU_STACK_GUARD")


class Kconfig(Struct):
	"""The options a Zephyr build's ``.config`` sets."""

	options: Mapping[str, str] = {}


NO_KCONFIG: Final = Kconfig()


def kconfig(text: str) -> Kconfig:
	r"""The options a ``.config`` sets, each with its value as written.

	>>> kconfig('CONFIG_IDLE_STACK_SIZE=256\n# CONFIG_FPU is not set\nCONFIG_BOARD="mps2"\n').options
	{'CONFIG_IDLE_STACK_SIZE': '256', 'CONFIG_BOARD': '"mps2"'}
	"""
	return Kconfig(options={option[1]: option[2] for option in _KCONFIG_OPTION.finditer(text)})


def _declared(options: Kconfig, stack_size: int | None) -> int | None:
	r"""A declared stack size, when the build's ``.config`` shows the thread can use all of it.

	Under ``native_sim`` threads run on host stacks, and an MPU stack guard can be carved out of
	the stack.

	>>> _declared(kconfig('CONFIG_ARCH="arm"\n'), 1024)
	1024
	>>> _declared(kconfig('CONFIG_ARCH="posix"\nCONFIG_ARCH_POSIX=y\n'), 1024) is None
	True
	>>> _declared(kconfig('CONFIG_ARCH="arm"\nCONFIG_MPU_STACK_GUARD=y\n'), 1024) is None
	True
	>>> _declared(kconfig("CONFIG_SMP=y\n"), 1024) is None
	True
	"""
	return (
		stack_size
		if _ARCH in options.options
		and all(options.options.get(option) != "y" for option in _STACK_NOT_ALL_USABLE)
		else None
	)


def _integer(options: Kconfig, name: str) -> int | None:
	r"""An option's value, when it is an integer.

	>>> _integer(kconfig("CONFIG_ISR_STACK_SIZE=0x800\nCONFIG_SMP=y\n"), "CONFIG_ISR_STACK_SIZE")
	2048
	>>> _integer(kconfig("CONFIG_SMP=y\n"), "CONFIG_SMP") is None
	True
	"""
	match options.options.get(name):
		case str() as value if _KCONFIG_INTEGER.fullmatch(value):
			return int(value, 0)
		case _:
			return None


_PRIORITY_BITS: Final = re.compile(r"arm,num-irq-priority-bits = < (0x[0-9a-f]+|[0-9]+) >;")


def detect(program: Program, options: Kconfig = NO_KCONFIG) -> RtosModel | None:
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
		thread
		for record in records
		if (thread := _thread(program, layout, record, options)) is not None
	)
	return RtosModel(
		name="zephyr",
		evidence=(_TRAMPOLINE, _RECORD),
		threads=threads,
		trampoline=_TRAMPOLINE,
		stack_reservation=_stack_reservation(program, options),
		exception_frame=_exception_frame(program.arm_core),
		system_threads=tuple(
			system_thread
			for name, entry, size in _SYSTEM_THREADS
			if (
				system_thread := _system_thread(
					program, name, entry, _declared(options, _integer(options, size))
				)
			)
			is not None
		),
		interrupt_stack=_interrupt_stack(
			program, _declared(options, _integer(options, _INTERRUPT_STACK_SIZE))
		),
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


def _system_thread(
	program: Program, name: str, entry: str, stack_size: int | None
) -> SystemThread | None:
	match tuple(
		address for address, function in program.functions.items() if function.name == entry
	):
		case (address,) if any(
			data_object.name == name for data_object in program.objects.values()
		):
			return SystemThread(name=name, entry=address, stack_size=stack_size)
		case _:
			return None


def _interrupt_stack(program: Program, stack_size: int | None) -> InterruptStack | None:
	match program.arm_core:
		case ArmCore(profile=ArmProfile.MICROCONTROLLER) if any(
			data_object.name == _INTERRUPT_STACK for data_object in program.objects.values()
		):
			return InterruptStack(name=_INTERRUPT_STACK, stack_size=stack_size)
		case ArmCore() | None:
			return None
		case _ as unreachable:
			assert_never(unreachable)


def priority_levels(devicetree: str) -> int | None:
	"""The priority levels an M-profile interrupt controller has, from Zephyr's ``zephyr.dts``.

	>>> priority_levels("arm,num-irq-priority-bits = < 0x3 >;")
	8
	>>> priority_levels("/ { };") is None
	True
	"""
	match _PRIORITY_BITS.search(devicetree):
		case None:
			return None
		case re.Match() as bits:
			return 1 << int(bits[1], 0)
		case _ as unreachable:
			assert_never(unreachable)


def _stack_reservation(program: Program, options: Kconfig) -> int:
	match program.machine:
		case Machine.EM_ARM:
			return _headroom(
				_tls_bytes(program, options) + _random_offset(options),
				_stack_pointer_alignment(options),
			)
		case Machine.EM_386 | Machine.EM_X86_64:
			return 0
		case _ as unreachable:
			assert_never(unreachable)


def _tls_bytes(program: Program, options: Kconfig) -> int:
	"""What an ARM thread's stack gives to thread-local storage (TLS) before its entry runs.

	The TLS copy and two toolchain pointers sit at the top of the stack. The ``.config`` says
	whether the build has TLS, and the setup function's symbol does without one or when
	link-time optimization did not inline it.
	"""
	return (
		program.tls_size + _TOOLCHAIN_TLS_POINTERS * program.pointer_size
		if _TLS_SETUP in program.symbol_addresses or options.options.get(_TLS) == "y"
		else 0
	)


def _random_offset(options: Kconfig) -> int:
	r"""The largest random offset Zephyr can lower a thread's first stack pointer by.

	>>> _random_offset(kconfig("CONFIG_STACK_POINTER_RANDOM=64\n"))
	63
	>>> _random_offset(NO_KCONFIG)
	0
	"""
	return max((_integer(options, _RANDOM_OFFSET) or 0) - 1, 0)


def _stack_pointer_alignment(options: Kconfig) -> int:
	r"""``ARCH_STACK_PTR_ALIGN`` on ARM: 8 with ``CONFIG_STACK_ALIGN_DOUBLE_WORD``, else 4.

	Without a ``.config``, the larger.

	>>> _stack_pointer_alignment(kconfig('CONFIG_ARCH="arm"\nCONFIG_STACK_ALIGN_DOUBLE_WORD=y\n'))
	8
	>>> _stack_pointer_alignment(kconfig('CONFIG_ARCH="arm"\n'))
	4
	>>> _stack_pointer_alignment(NO_KCONFIG)
	8
	"""
	return (
		_DOUBLE_WORD
		if _ARCH not in options.options or options.options.get(_ALIGN_DOUBLE_WORD) == "y"
		else _SINGLE_WORD
	)


def _headroom(delta: int, alignment: int) -> int:
	"""The top of a thread's stack Zephyr takes, rounded up to the stack pointer's alignment.

	>>> _headroom(4 + 2 * 4, 8)
	16
	>>> _headroom(4 + 2 * 4, 4)
	12
	>>> _headroom(0, 8)
	0
	"""
	return (delta + alignment - 1) // alignment * alignment


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


def _thread(
	program: Program, layout: StructureLayout, record: DataObject, options: Kconfig
) -> ThreadRoot | None:
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
				stack_size=_declared(
					options, _word(program, record, layout.offsets.get(_STACK_SIZE))
				),
			)
		case _:
			return None


def _word(program: Program, record: DataObject, offset: int | None) -> Address | None:
	return pointer_at(program, Address(record.address + offset)) if offset is not None else None
