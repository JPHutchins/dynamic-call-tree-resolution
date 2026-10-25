# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""What Zephyr adds to a stack beyond its deepest path, and the stacks beside its threads'."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final, assert_never

from salix import Struct

from dynamic_call_tree_resolution.model import (
	Address,
	ArmCore,
	ArmProfile,
	Interrupt,
	InterruptStack,
	Machine,
	PrivilegedStack,
	Uncounted,
)
from dynamic_call_tree_resolution.points_to import in_writable_memory, pointer_at
from dynamic_call_tree_resolution.rtos.zephyr.options import (
	Kconfig,
	configured,
	random_offset,
	stack_pointer_alignment,
)

if TYPE_CHECKING:
	from dynamic_call_tree_resolution.model import Program


_BASIC_FRAME: Final = 8 * 4
_EXTENDED_FRAME: Final = 26 * 4
_ALIGNMENT_PAD: Final = 4
_TLS_SETUP: Final = "arch_tls_stack_setup"
_TOOLCHAIN_TLS_POINTERS: Final = 2
_TLS: Final = "CONFIG_THREAD_LOCAL_STORAGE"
_LOCAL_DATA: Final = "CONFIG_THREAD_USERSPACE_LOCAL_DATA"
_LOCAL_DATA_TYPE: Final = "struct _thread_userspace_local_data"
_LOCAL_DATA_BYTES: Final = 4
_SENTINEL: Final = "CONFIG_STACK_SENTINEL"
_SENTINEL_CHECK: Final = "z_check_stack_sentinel"
_SENTINEL_BYTES: Final = 4
_SWITCH: Final = "CONFIG_USE_SWITCH"
_SWITCH_CODE: Final = ("arm_m_exc_exit", "z_arm_context_switch")
_SYSTEM_MODE_FRAME: Final = (2 + 6) * 4
_VFP_FRAME: Final = (16 + 2) * 4
_VFP_D32_FRAME: Final = _VFP_FRAME + 16 * 8
_FPU_SHARING: Final = "CONFIG_FPU_SHARING"
_D32: Final = "CONFIG_VFP_FEATURE_REGS_S64_D32"
_DYNAMIC_INTERRUPTS: Final = "CONFIG_DYNAMIC_INTERRUPTS"
_ISR_TABLE: Final = "_sw_isr_table"
_ISR_ENTRY: Final = "struct _isr_table_entry"
_ISR: Final = "isr"
_SPURIOUS: Final = "z_irq_spurious"
_INTERRUPT_STACK: Final = "z_interrupt_stacks"
INTERRUPT_STACK_SIZE: Final = "CONFIG_ISR_STACK_SIZE"
_PRIVILEGED_STACKS: Final = "priv_stacks"
PRIVILEGED_STACK_SIZE: Final = "CONFIG_PRIVILEGED_STACK_SIZE"
_SYSCALL_ENTRY: Final = "z_arm_do_syscall"
_USER_START: Final = "arch_user_mode_enter"
_SYSCALL_SAVED_WORDS: Final = 2
_PRIORITY_BITS: Final = re.compile(r"arm,num-irq-priority-bits = < (?P<bits>0x[0-9a-f]+|[0-9]+) >;")


def stack_reservation(program: Program, options: Kconfig) -> int:
	match program.machine:
		case Machine.EM_ARM:
			return _headroom(
				_tls_bytes(program, options)
				+ _local_data_bytes(program, options)
				+ random_offset(options),
				stack_pointer_alignment(options),
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


def _local_data_bytes(program: Program, options: Kconfig) -> int:
	"""The userspace-local data Zephyr keeps below TLS when errno lives outside it."""
	return (
		_LOCAL_DATA_BYTES
		if options.options.get(_LOCAL_DATA) == "y" or _LOCAL_DATA_TYPE in program.layouts
		else 0
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


def exception_frame(core: ArmCore | None, options: Kconfig) -> int:
	"""What one interrupt stacks on the thread's stack it takes.

	ARMv7-M stacks a basic frame of 8 words, or 26 with a floating-point context, and may
	pad a word to align the stack to 8 bytes. On an A- or R-profile core Zephyr's interrupt
	entry stores the return state and 6 registers on the thread's stack in system mode, and
	reserves the floating-point frame Zephyr shares there.

	>>> exception_frame(ArmCore(profile=ArmProfile.MICROCONTROLLER, floating_point=False), Kconfig())
	36
	>>> exception_frame(ArmCore(profile=ArmProfile.MICROCONTROLLER, floating_point=True), Kconfig())
	108
	>>> exception_frame(ArmCore(profile=ArmProfile.REAL_TIME, floating_point=False), Kconfig())
	32
	>>> exception_frame(None, Kconfig())
	0
	"""
	match core:
		case ArmCore(profile=ArmProfile.MICROCONTROLLER, floating_point=floating_point):
			return (_EXTENDED_FRAME if floating_point else _BASIC_FRAME) + _ALIGNMENT_PAD
		case ArmCore(floating_point=floating_point):
			return _SYSTEM_MODE_FRAME + _floating_point_frame(floating_point, options)
		case None:
			return 0
		case _ as unreachable:
			assert_never(unreachable)


def _floating_point_frame(floating_point: bool, options: Kconfig) -> int:
	"""What an A- or R-profile interrupt entry reserves for floating-point state.

	Without a ``.config``, the most the build attributes allow.

	>>> _floating_point_frame(True, Kconfig()), _floating_point_frame(False, Kconfig())
	(200, 0)
	"""
	return (
		(_VFP_D32_FRAME if floating_point else 0)
		if not configured(options)
		else (_VFP_D32_FRAME if options.options.get(_D32) == "y" else _VFP_FRAME)
		if options.options.get(_FPU_SHARING) == "y"
		else 0
	)


def exception_frames(program: Program, options: Kconfig) -> int | None:
	"""How many frames can stack on a thread's stack at once; none when unknown.

	On M-profile, handlers run on the main stack. On an A- or R-profile core each nested
	interrupt stacks its own frame on the thread's stack, and an interrupt nests only while it
	is not active already: one for each interrupt the image connects, one for an unconnected
	line's fatal handler, and one for a system call a handler makes.
	"""
	match program.arm_core:
		case ArmCore(profile=ArmProfile.MICROCONTROLLER):
			return 1
		case ArmCore() | None:
			match interrupt_table(program, options):
				case InterruptTable(connected=connected, spurious=spurious):
					return len(connected) + len(spurious) + 1
				case None:
					return None
				case _ as unreachable:
					assert_never(unreachable)
		case _ as unreachable:
			assert_never(unreachable)


class InterruptTable(Struct):
	"""The interrupts an image's table connects, and one line it leaves unconnected."""

	connected: tuple[Interrupt, ...]
	spurious: tuple[Interrupt, ...]


def interrupt_table(program: Program, options: Kconfig) -> InterruptTable | None:
	"""The interrupts that can nest on an A- or R-profile core; none when they can't be told."""
	table = next(
		(data_object for data_object in program.objects.values() if data_object.name == _ISR_TABLE),
		None,
	)
	layout = program.layouts.get(_ISR_ENTRY)
	match program.arm_core:
		case ArmCore(profile=ArmProfile.APPLICATION | ArmProfile.REAL_TIME) if not (
			switches_on_stack(program, options)
			or options.options.get(_DYNAMIC_INTERRUPTS) == "y"
			or table is None
			or layout is None
			or _ISR not in layout.offsets
			or layout.size == 0
			or in_writable_memory(program, table.address)
		):
			return _interrupts(
				program,
				tuple(
					pointer_at(program, Address(table.address + offset))
					for offset in range(layout.offsets[_ISR], table.size, layout.size)
				),
			)
		case ArmCore() | None:
			return None
		case _ as unreachable:
			assert_never(unreachable)


def _interrupts(program: Program, handlers: tuple[Address | None, ...]) -> InterruptTable | None:
	interrupts = tuple(
		Interrupt(number=number, handler=handler)
		for number, handler in enumerate(handlers)
		if handler is not None
	)
	spurious = addresses_of(program, _SPURIOUS)
	return (
		InterruptTable(
			connected=tuple(
				interrupt for interrupt in interrupts if interrupt.handler not in spurious
			),
			spurious=tuple(interrupt for interrupt in interrupts if interrupt.handler in spurious)[
				:1
			],
		)
		if len(interrupts) == len(handlers)
		else None
	)


def stack_sentinel(program: Program, options: Kconfig) -> int:
	"""The word Zephyr writes at the bottom of each thread's stack and checks for an overflow."""
	return (
		_SENTINEL_BYTES
		if options.options.get(_SENTINEL) == "y" or _SENTINEL_CHECK in program.symbol_addresses
		else 0
	)


def not_counted(program: Program, options: Kconfig) -> frozenset[Uncounted]:
	"""What the image shows Zephyr adds to a thread's stack that the model leaves out."""
	return frozenset(
		uncounted
		for uncounted, shown in {
			Uncounted.EXCEPTION_FRAMES: program.machine is Machine.EM_ARM
			and exception_frames(program, options) is None,
			Uncounted.SWITCH_FRAME: switches_on_stack(program, options),
		}.items()
		if shown
	)


def switches_on_stack(program: Program, options: Kconfig) -> bool:
	return options.options.get(_SWITCH) == "y" or any(
		symbol in program.symbol_addresses for symbol in _SWITCH_CODE
	)


def interrupt_stack(program: Program, stack_size: int | None) -> InterruptStack | None:
	match program.arm_core:
		case ArmCore(profile=ArmProfile.MICROCONTROLLER) if any(
			data_object.name == _INTERRUPT_STACK for data_object in program.objects.values()
		):
			return InterruptStack(name=_INTERRUPT_STACK, stack_size=stack_size)
		case ArmCore() | None:
			return None
		case _ as unreachable:
			assert_never(unreachable)


def privileged_stack(program: Program, stack_size: int | None) -> PrivilegedStack | None:
	"""Where an M-profile core starts a user thread and runs its system calls.

	The SVC handler leaves the caller's stack pointer and return address at the top.
	"""
	match (
		program.arm_core,
		addresses_of(program, _SYSCALL_ENTRY),
		addresses_of(program, _USER_START),
	):
		case ArmCore(profile=ArmProfile.MICROCONTROLLER), (system_call,), (user_start,) if any(
			data_object.name == _PRIVILEGED_STACKS for data_object in program.objects.values()
		):
			return PrivilegedStack(
				name=_PRIVILEGED_STACKS,
				system_call=system_call,
				user_start=user_start,
				saved=_SYSCALL_SAVED_WORDS * program.pointer_size,
				stack_size=stack_size,
			)
		case _:
			return None


def addresses_of(program: Program, name: str) -> tuple[Address, ...]:
	return tuple(
		address for address, function in program.functions.items() if function.name == name
	)


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
			return 1 << int(bits["bits"], 0)
		case _ as unreachable:
			assert_never(unreachable)
