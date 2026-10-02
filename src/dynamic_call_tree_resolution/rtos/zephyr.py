# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Zephyr's static threads, from the ``K_THREAD_DEFINE`` records in the image."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from dynamic_call_tree_resolution.model import Address, RtosModel, ThreadCreation, ThreadRoot
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
