# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The stacks an A- or R-profile core switches to in its processor modes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, assert_never

from salix import Struct

from dynamic_call_tree_resolution.model import (
	ArmCore,
	ArmProfile,
	HandlerStack,
	ModeStack,
	Uncounted,
)
from dynamic_call_tree_resolution.rtos.zephyr.options import (
	Kconfig,
	configured,
	declared,
	integer,
)
from dynamic_call_tree_resolution.rtos.zephyr.stacks import (
	InterruptTable,
	addresses_of,
	exception_frame,
	interrupt_table,
	switches_on_stack,
)

if TYPE_CHECKING:
	from dynamic_call_tree_resolution.model import Address, Program


def mode_stacks(program: Program, options: Kconfig) -> tuple[ModeStack, ...]:
	match program.arm_core:
		case ArmCore(profile=ArmProfile.APPLICATION | ArmProfile.REAL_TIME) if not (
			switches_on_stack(program, options)
		):
			return tuple(
				stack
				for spec in _MODE_STACKS
				if (stack := _mode_stack(program, options, spec)) is not None
			)
		case ArmCore() | None:
			return ()
		case _ as unreachable:
			assert_never(unreachable)


class _ModeStackSpec(Struct):
	"""A mode's stack, the option that sizes it, and the functions its exceptions start on it."""

	name: str
	size_option: str
	entries: tuple[str, ...]
	exception_entry: bool
	"""Whether the mode's exception entry stacks a frame on it before those functions run."""


_MODE_STACKS: Final = (
	_ModeStackSpec(
		name="z_arm_sys_stack",
		size_option="CONFIG_ARMV7_SYS_STACK_SIZE",
		entries=(
			"soc_reset_hook",
			"z_arm_watchdog_init",
			"z_arm_tcm_disable_ecc",
			"z_arm_relocate_vector_table",
			"z_prep_c",
		),
		exception_entry=False,
	),
	_ModeStackSpec(
		name="z_arm_fiq_stack",
		size_option="CONFIG_ARMV7_FIQ_STACK_SIZE",
		entries=("z_arm_nmi",),
		exception_entry=False,
	),
	_ModeStackSpec(
		name="z_arm_abort_stack",
		size_option="CONFIG_ARMV7_EXCEPTION_STACK_SIZE",
		entries=("z_arm_fault_data", "z_arm_fault_prefetch"),
		exception_entry=True,
	),
	_ModeStackSpec(
		name="z_arm_undef_stack",
		size_option="CONFIG_ARMV7_EXCEPTION_STACK_SIZE",
		entries=("z_arm_fault_undef_instruction", "z_arm_fault_undef_instruction_fp"),
		exception_entry=True,
	),
)
_EXTRA_INFO: Final = "CONFIG_EXTRA_EXCEPTION_INFO"
_EXTRA_INFO_TYPE: Final = "struct __extra_esf_info"


def _mode_stack(program: Program, options: Kconfig, spec: _ModeStackSpec) -> ModeStack | None:
	entries = _present(program, spec.entries)
	return (
		ModeStack(
			name=spec.name,
			entries=entries,
			exception_frame=exception_frame(program.arm_core, options)
			if spec.exception_entry
			else 0,
			not_counted=(
				frozenset({Uncounted.EXCEPTION_FRAMES})
				if spec.exception_entry and _extra_info(program, options)
				else frozenset()
			),
			stack_size=declared(options, integer(options, spec.size_option)),
		)
		if entries
		and any(data_object.name == spec.name for data_object in program.objects.values())
		else None
	)


def _extra_info(program: Program, options: Kconfig) -> bool:
	return options.options.get(_EXTRA_INFO) == "y" or _EXTRA_INFO_TYPE in program.layouts


def handler_stack(program: Program, options: Kconfig) -> HandlerStack | None:
	"""Where an A- or R-profile core runs interrupt handlers and system calls: SVC mode's stack.

	Each nested interrupt's entry stacks the return address, an alignment word and two
	register pairs there, or another pair to track the active interrupt; a system call's
	entry, the return address, an alignment word and one pair.
	"""
	match interrupt_table(program, options):
		case InterruptTable() as table if any(
			data_object.name == _HANDLER_STACK for data_object in program.objects.values()
		):
			return HandlerStack(
				name=_HANDLER_STACK,
				interrupts=table.connected,
				spurious=table.spurious,
				wrapper_calls=_present(program, _WRAPPER_CALLS),
				interrupt_frame=_INTERRUPT_ENTRY
				+ (
					_TRACKED_ENTRY
					if not configured(options) or options.options.get(_TRACK_ACTIVE) == "y"
					else 0
				),
				system_calls=_present(program, _SYSTEM_CALLS),
				system_call_frame=_SYSTEM_CALL_ENTRY,
				stack_size=declared(options, integer(options, _HANDLER_STACK_SIZE)),
			)
		case InterruptTable() | None:
			return None
		case _ as unreachable:
			assert_never(unreachable)


_HANDLER_STACK: Final = "z_arm_svc_stack"
_HANDLER_STACK_SIZE: Final = "CONFIG_ARMV7_SVC_STACK_SIZE"
_WRAPPER_CALLS: Final = (
	"arm_gic_get_active",
	"z_soc_irq_get_active",
	"arm_gic_eoi",
	"z_soc_irq_eoi",
	"sys_trace_isr_enter",
	"sys_trace_isr_exit",
	"pm_system_resume",
)
_SYSTEM_CALLS: Final = ("z_arm_do_swap", "z_do_kernel_oops", "z_irq_do_offload")
_INTERRUPT_ENTRY: Final = (1 + 1 + 2 + 2) * 4
_TRACKED_ENTRY: Final = 2 * 4
_TRACK_ACTIVE: Final = "CONFIG_ARM_TRACK_ACTIVE_IRQ"
_SYSTEM_CALL_ENTRY: Final = (1 + 1 + 2) * 4


def _present(program: Program, names: tuple[str, ...]) -> tuple[Address, ...]:
	return tuple(address for name in names for address in addresses_of(program, name))
