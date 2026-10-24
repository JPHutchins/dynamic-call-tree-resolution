# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The calls Zephyr makes through each entry of a linker-built table or array."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, assert_never

from salix import Struct

from dynamic_call_tree_resolution.model import (
	Address,
	Dispatch,
	EmbeddedStructMember,
)
from dynamic_call_tree_resolution.points_to import in_writable_memory, pointer_at

if TYPE_CHECKING:
	from dynamic_call_tree_resolution.model import Program


class _SectionDispatch(Struct):
	"""A function that calls through a member of each entry of a linker-built table."""

	function: str
	type_name: str
	member: tuple[str, ...]
	"""The path to the called member, through embedded structs."""
	start_label: str
	end_label: str


class _ArrayDispatch(Struct):
	"""A function that calls through an element of an array of function pointers."""

	function: str
	array: str


type _DispatchSpec = _SectionDispatch | _ArrayDispatch


DISPATCHES: Final[tuple[_DispatchSpec, ...]] = (
	_SectionDispatch(
		function="z_sys_init_run_level",
		type_name="struct init_entry",
		member=("init_fn",),
		start_label="__init_start",
		end_label="__init_end",
	),
	_SectionDispatch(
		function="do_device_init",
		type_name="struct device",
		member=("ops", "init"),
		start_label="_device_list_start",
		end_label="_device_list_end",
	),
	_ArrayDispatch(function="z_arm_do_syscall", array="_k_syscall_table"),
)


def dispatch_of(program: Program, spec: _DispatchSpec) -> Dispatch | None:
	match spec:
		case _SectionDispatch():
			return _section_dispatch(program, spec)
		case _ArrayDispatch():
			return _array_dispatch(program, spec)
		case _ as unreachable:
			assert_never(unreachable)


def _section_dispatch(program: Program, spec: _SectionDispatch) -> Dispatch | None:
	entries = tuple(
		data_object
		for data_object in program.objects.values()
		if data_object.type_name == spec.type_name
	)
	offset = _member_offset(program, spec.type_name, spec.member)
	start = program.labels.get(spec.start_label)
	end = program.labels.get(spec.end_label)
	if (
		spec.function not in program.symbol_addresses
		or offset is None
		or start is None
		or end is None
		or not entries
		or not all(
			start <= entry.address < end and not in_writable_memory(program, entry.address)
			for entry in entries
		)
	):
		return None
	return Dispatch(
		function=spec.function,
		targets=frozenset(
			target
			for entry in entries
			for target in (pointer_at(program, Address(entry.address + offset)),)
			if target is not None and target in program.functions
		),
	)


def _array_dispatch(program: Program, spec: _ArrayDispatch) -> Dispatch | None:
	array = next(
		(data_object for data_object in program.objects.values() if data_object.name == spec.array),
		None,
	)
	if (
		array is None
		or spec.function not in program.symbol_addresses
		or in_writable_memory(program, array.address)
	):
		return None
	slots = tuple(
		pointer_at(program, Address(array.address + offset))
		for offset in range(0, array.size, program.pointer_size)
	)
	return (
		Dispatch(
			function=spec.function,
			targets=frozenset(target for target in slots if target is not None),
		)
		if slots and all(target is not None and target in program.functions for target in slots)
		else None
	)


def _member_offset(program: Program, type_name: str, member: tuple[str, ...]) -> int | None:
	layout = program.layouts.get(type_name)
	if layout is None or member[0] not in layout.offsets:
		return None
	if len(member) == 1:
		return layout.offsets[member[0]]
	nested = next(
		(
			embedded.type_name
			for embedded in layout.members
			if isinstance(embedded, EmbeddedStructMember) and embedded.name == member[0]
		),
		None,
	)
	inner = _member_offset(program, nested, member[1:]) if nested is not None else None
	return layout.offsets[member[0]] + inner if inner is not None else None
