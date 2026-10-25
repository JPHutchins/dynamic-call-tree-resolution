# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""What Zephyr adds to a stack beyond its deepest path, and the stacks beside its threads'."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final, assert_never

from dynamic_call_tree_resolution.model import (
	Address,
	ArmCore,
	ArmProfile,
	InterruptStack,
	Machine,
	PrivilegedStack,
	Uncounted,
)
from dynamic_call_tree_resolution.rtos.zephyr.options import (
	Kconfig,
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


def exception_frame(core: ArmCore | None) -> int:
	"""What an M-profile core stacks on the thread's stack when an interrupt takes it.

	ARMv7-M stacks a basic frame of 8 words, or 26 with a floating-point context, and may
	pad a word to align the stack to 8 bytes. Handlers then run on the main stack, so a
	nested interrupt adds nothing more. Other profiles are not modeled.

	>>> exception_frame(ArmCore(profile=ArmProfile.MICROCONTROLLER, floating_point=False))
	36
	>>> exception_frame(ArmCore(profile=ArmProfile.MICROCONTROLLER, floating_point=True))
	108
	>>> exception_frame(ArmCore(profile=ArmProfile.APPLICATION, floating_point=True))
	0
	"""
	match core:
		case ArmCore(profile=ArmProfile.MICROCONTROLLER, floating_point=floating_point):
			return (_EXTENDED_FRAME if floating_point else _BASIC_FRAME) + _ALIGNMENT_PAD
		case ArmCore() | None:
			return 0
		case _ as unreachable:
			assert_never(unreachable)


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
			and exception_frame(program.arm_core) == 0,
			Uncounted.SWITCH_FRAME: options.options.get(_SWITCH) == "y"
			or any(symbol in program.symbol_addresses for symbol in _SWITCH_CODE),
		}.items()
		if shown
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
		_addresses_of(program, _SYSCALL_ENTRY),
		_addresses_of(program, _USER_START),
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


def _addresses_of(program: Program, name: str) -> tuple[Address, ...]:
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
