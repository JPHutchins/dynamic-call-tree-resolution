# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Thread creation, as the analysis models it."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from salix import Struct, replace

from dynamic_call_tree_resolution.model import (
	FUNCTION_POINTER,
	Address,
	Machine,
)
from dynamic_call_tree_resolution.span import Span
from dynamic_call_tree_resolution.vsa.abi import (
	normalized,
)
from dynamic_call_tree_resolution.vsa.interpret import (
	CallObservation,
)
from dynamic_call_tree_resolution.vsa.lattice import Known, Top, ValueSet, meet
from dynamic_call_tree_resolution.vsa.links import function_covering

if TYPE_CHECKING:
	from collections.abc import Collection, Mapping

	from dynamic_call_tree_resolution.model import (
		Function,
		Program,
		RtosModel,
		ThreadCreation,
	)


class Creation(Struct):
	"""Thread creation, gated so that the trampoline is entered only from its builders' frames."""

	trampoline: Address
	entry_positions: Mapping[Address, int]
	"""Each frame builder's function-pointer argument, by the builder's start."""
	setup: Address
	setup_entry: int
	static_spans: tuple[Span, ...]
	static_entries: ValueSet


def thread_creation(
	program: Program,
	rtos: RtosModel,
	taken: frozenset[Address],
	references: Mapping[Address, frozenset[Address]],
) -> Creation | None:
	match rtos.creation, rtos.trampoline:
		case None, _:
			return None
		case _, None:
			return None  # pragma: no cover
		case creation, str() as trampoline_name:
			return _gated_creation(program, creation, trampoline_name, taken, references)
		case _ as unreachable:
			assert_never(unreachable)


def _gated_creation(
	program: Program,
	creation: ThreadCreation,
	trampoline_name: str,
	taken: frozenset[Address],
	references: Mapping[Address, frozenset[Address]],
) -> Creation | None:
	builders = {
		builder: _entry_position(program.functions[builder])
		for name in creation.frame_builders
		for builder in functions_named(program, name)
	}
	match functions_named(program, trampoline_name), functions_named(program, creation.setup):
		case (trampoline,), (setup,) if (
			builders
			and None not in builders.values()
			and not builders.keys() & taken
			and _entry_position(program.functions[trampoline]) == 0
			and (setup_entry := _entry_position(program.functions[setup])) is not None
			and _held_only_by(program, references.get(trampoline, frozenset()), builders.keys())
		):
			return Creation(
				trampoline=normalized(trampoline, program.machine),
				entry_positions={
					normalized(builder, program.machine): position
					for builder, position in builders.items()
					if position is not None
				},
				setup=normalized(setup, program.machine),
				setup_entry=setup_entry,
				static_spans=(
					*program.inlined.get(creation.static_start, ()),
					*(
						_span(program.functions[function], program.machine)
						for function in functions_named(program, creation.static_start)
					),
				),
				static_entries=(
					Known(values=creation.static_entries)
					if creation.static_entries is not None
					else Top()
				),
			)
		case _:
			return None


def functions_named(program: Program, name: str) -> tuple[Address, ...]:
	return tuple(
		address for address, function in program.functions.items() if function.name == name
	)


def _entry_position(function: Function) -> int | None:
	match tuple(
		position
		for position, parameter in enumerate(
			function.signature.parameters if function.signature is not None else ()
		)
		if parameter == FUNCTION_POINTER
	):
		case (position,):
			return position
		case _:
			return None


def _span(function: Function, machine: Machine) -> Span:
	start = normalized(function.address, machine)
	return Span(start=start, end=Address(start + function.size))


def _held_only_by(
	program: Program, slots: frozenset[Address], holders: Collection[Address]
) -> bool:
	return bool(slots) and all(
		(function := function_covering(program, slot)) is not None and function.address in holders
		for slot in slots
	)


def modeled(
	observations: list[CallObservation], creation: Creation | None
) -> list[CallObservation]:
	match creation:
		case None:
			return observations
		case Creation():
			substituted = [_static_entries(observation, creation) for observation in observations]
			return [
				*substituted,
				*(
					CallObservation(
						callee=creation.trampoline,
						site=observation.site,
						arguments={
							0: observation.arguments.get(
								creation.entry_positions[observation.callee], Top()
							)
						},
					)
					for observation in substituted
					if observation.callee in creation.entry_positions
				),
			]
		case _ as unreachable:
			assert_never(unreachable)


def _static_entries(observation: CallObservation, creation: Creation) -> CallObservation:
	return (
		replace(
			observation,
			arguments={
				**observation.arguments,
				creation.setup_entry: meet(
					observation.arguments.get(creation.setup_entry, Top()), creation.static_entries
				),
			},
		)
		if observation.callee == creation.setup
		and any(observation.site in span for span in creation.static_spans)
		else observation
	)
