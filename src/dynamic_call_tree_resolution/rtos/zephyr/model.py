# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The Zephyr model of an image."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dynamic_call_tree_resolution.model import (
	RtosModel,
	ThreadCreation,
)
from dynamic_call_tree_resolution.rtos.zephyr.dispatch import (
	DISPATCHES,
	dispatch_of,
)
from dynamic_call_tree_resolution.rtos.zephyr.modes import mode_stacks
from dynamic_call_tree_resolution.rtos.zephyr.options import (
	NO_KCONFIG,
	Kconfig,
	declared,
	integer,
)
from dynamic_call_tree_resolution.rtos.zephyr.stacks import (
	INTERRUPT_STACK_SIZE,
	PRIVILEGED_STACK_SIZE,
	exception_frame,
	exception_frames,
	interrupt_stack,
	not_counted,
	privileged_stack,
	stack_reservation,
	stack_sentinel,
)
from dynamic_call_tree_resolution.rtos.zephyr.threads import (
	FRAME_BUILDERS,
	RECORD,
	SETUP,
	STATIC_START,
	SYSTEM_THREADS,
	TRAMPOLINE,
	address_of,
	system_thread_of,
	thread_of,
)

if TYPE_CHECKING:
	from dynamic_call_tree_resolution.model import Program


def detect(program: Program, options: Kconfig = NO_KCONFIG) -> RtosModel | None:
	layout = program.layouts.get(RECORD)
	if layout is None or all(
		function.name != TRAMPOLINE for function in program.functions.values()
	):
		return None
	records = tuple(
		record
		for record in sorted(program.objects.values(), key=address_of)
		if record.type_name == RECORD
	)
	threads = tuple(
		thread
		for record in records
		if (thread := thread_of(program, layout, record, options)) is not None
	)
	return RtosModel(
		name="zephyr",
		evidence=(TRAMPOLINE, RECORD),
		threads=threads,
		trampoline=TRAMPOLINE,
		stack_reservation=stack_reservation(program, options),
		exception_frame=exception_frame(program.arm_core, options),
		exception_frames=frames
		if (frames := exception_frames(program, options)) is not None
		else 1,
		stack_sentinel=stack_sentinel(program, options),
		system_threads=tuple(
			system_thread
			for spec in SYSTEM_THREADS
			if (
				system_thread := system_thread_of(
					program,
					spec.name,
					spec.entry,
					declared(options, integer(options, spec.stack_size_option)),
				)
			)
			is not None
		),
		interrupt_stack=interrupt_stack(
			program, declared(options, integer(options, INTERRUPT_STACK_SIZE))
		),
		mode_stacks=mode_stacks(program, options),
		privileged_stack=privileged_stack(
			program, declared(options, integer(options, PRIVILEGED_STACK_SIZE))
		),
		dispatches=tuple(
			dispatch for spec in DISPATCHES if (dispatch := dispatch_of(program, spec)) is not None
		),
		creation=ThreadCreation(
			frame_builders=FRAME_BUILDERS,
			setup=SETUP,
			static_start=STATIC_START,
			static_entries=(
				frozenset(thread.entry for thread in threads)
				if len(threads) == len(records)
				else None
			),
		),
		not_counted=not_counted(program, options),
	)
