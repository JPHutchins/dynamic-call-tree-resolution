# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Zephyr's static threads, from the ``K_THREAD_DEFINE`` records in the image."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from salix import Struct

from dynamic_call_tree_resolution.model import (
	Address,
	SystemThread,
	ThreadRoot,
)
from dynamic_call_tree_resolution.points_to import in_writable_memory, pointer_at
from dynamic_call_tree_resolution.rtos.zephyr.options import (
	Kconfig,
	declared,
)

if TYPE_CHECKING:
	from dynamic_call_tree_resolution.model import DataObject, Program, StructureLayout


RECORD: Final = "struct _static_thread_data"
TRAMPOLINE: Final = "z_thread_entry"
_RECORD_PREFIX: Final = "_k_thread_data_"
_ENTRY: Final = "init_entry"
_ARGUMENTS: Final = ("init_p1", "init_p2", "init_p3")
_STACK_SIZE: Final = "init_stack_size"
FRAME_BUILDERS: Final = ("arch_new_thread", "arch_switch_to_main_thread")
SETUP: Final = "z_setup_new_thread"
STATIC_START: Final = "z_init_static_threads"


class _SystemThreadSpec(Struct):
	"""A thread Zephyr creates while it starts, and the option that sizes its stack."""

	name: str
	entry: str
	stack_size_option: str


SYSTEM_THREADS: Final = (
	_SystemThreadSpec(
		name="z_main_thread", entry="bg_thread_main", stack_size_option="CONFIG_MAIN_STACK_SIZE"
	),
	_SystemThreadSpec(
		name="z_idle_threads", entry="idle", stack_size_option="CONFIG_IDLE_STACK_SIZE"
	),
)


def thread_of(
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
				stack_size=declared(
					options, _word(program, record, layout.offsets.get(_STACK_SIZE))
				),
			)
		case _:
			return None


def system_thread_of(
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


def address_of(data_object: DataObject) -> Address:
	return data_object.address


def _word(program: Program, record: DataObject, offset: int | None) -> Address | None:
	return pointer_at(program, Address(record.address + offset)) if offset is not None else None
