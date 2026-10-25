# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The stacks an A- or R-profile core switches to in its processor modes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, assert_never

from salix import Struct

from dynamic_call_tree_resolution.model import ArmCore, ArmProfile, ModeStack, Uncounted
from dynamic_call_tree_resolution.rtos.zephyr.options import Kconfig, declared, integer
from dynamic_call_tree_resolution.rtos.zephyr.stacks import (
	addresses_of,
	exception_frame,
	switches_on_stack,
)

if TYPE_CHECKING:
	from dynamic_call_tree_resolution.model import Program


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
	entries = tuple(address for name in spec.entries for address in addresses_of(program, name))
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
