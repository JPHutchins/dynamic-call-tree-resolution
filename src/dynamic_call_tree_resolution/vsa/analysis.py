# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Interprocedural rounds over every function's interpretation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Final, assert_never, cast

from capstone import Cs

from dynamic_call_tree_resolution.model import Address, CallSite, Machine
from dynamic_call_tree_resolution.vsa.abi import (
	ARM_ARGUMENT_REGISTERS,
	DISASSEMBLERS,
	SP_REGISTERS,
	X86_64_ARGUMENT_REGISTERS,
	normalized,
)
from dynamic_call_tree_resolution.vsa.cfg import Block, control_flow_graphs
from dynamic_call_tree_resolution.vsa.interpret import (
	CallObservation,
	FunctionResult,
	analyze_function,
)
from dynamic_call_tree_resolution.vsa.lattice import Known, Top
from dynamic_call_tree_resolution.vsa.memory import Context, accumulate_writes, context_for
from dynamic_call_tree_resolution.vsa.state import State, entry_state, join_seeds

if TYPE_CHECKING:
	from collections.abc import Mapping

	from capstone import ArmCsOperand

	from dynamic_call_tree_resolution.model import Function, Program

_GLOBAL_ROUNDS: Final = 3

_MAX_ROUNDS: Final = 8


def analyze(program: Program) -> tuple[CallSite, ...]:
	disassembler = Cs(*DISASSEMBLERS[program.machine])
	disassembler.detail = True
	disassembler.skipdata = True
	blocks_by_function = control_flow_graphs(program, disassembler)
	functions = tuple(blocks_by_function.values())
	_prewarm_instructions(functions, program.machine)
	seeds: dict[Address, State] = {}
	global_writes: dict[Address, frozenset[Address]] = {}
	stable_write_rounds = 0
	with ThreadPoolExecutor() as executor:
		for _ in range(_MAX_ROUNDS):
			context = context_for(program, global_writes)
			observations: list[CallObservation] = []
			written: list[Mapping[Address, frozenset[Address]]] = []
			for result in executor.map(
				partial(_round_analysis, program, context, seeds), functions
			):
				written.append(result.writes)
				observations.extend(result.observations)
			next_seeds = dict(seeds)
			for observation in observations:
				next_seeds[observation.callee] = join_seeds(
					next_seeds.get(observation.callee),
					_seed_from_observation(observation, program),
				)
			next_writes = accumulate_writes(global_writes, written)
			if next_seeds == seeds and next_writes == global_writes:
				break
			seeds_changed = next_seeds != seeds
			seeds, global_writes = next_seeds, next_writes
			if seeds_changed:
				stable_write_rounds = 0
			else:
				stable_write_rounds += 1
				if stable_write_rounds >= _GLOBAL_ROUNDS:
					break
		final_context = context_for(program, global_writes)
		return tuple(
			site
			for sites in executor.map(
				partial(_final_sites, program, final_context, seeds), functions
			)
			for site in sites
		)


def _round_analysis(
	program: Program,
	context: Context,
	seeds: Mapping[Address, State],
	function_blocks: tuple[Function, tuple[Block, ...]],
) -> FunctionResult:
	function, blocks = function_blocks
	seed = entry_state(seeds.get(normalized(function.address, program.machine)), program.machine)
	return analyze_function(context, function, blocks, seed)


def _final_sites(
	program: Program,
	context: Context,
	seeds: Mapping[Address, State],
	function_blocks: tuple[Function, tuple[Block, ...]],
) -> tuple[CallSite, ...]:
	function, blocks = function_blocks
	return analyze_function(
		context,
		function,
		blocks,
		entry_state(seeds.get(normalized(function.address, program.machine)), program.machine),
	).sites


def _prewarm_instructions(
	functions: tuple[tuple[Function, tuple[Block, ...]], ...], machine: Machine
) -> None:
	for _, blocks in functions:
		for block in blocks:
			for instruction in block.instructions:
				if instruction.mnemonic == ".byte":
					continue
				for operand in instruction.operands:
					if machine is Machine.EM_ARM:
						_ = cast("ArmCsOperand", operand).shift
					else:
						_ = operand.mem


def _seed_from_observation(observation: CallObservation, program: Program) -> State:
	registers: dict[int, frozenset[Address]] = {}
	stack: dict[int, frozenset[Address]] = {}
	for position, value in observation.arguments.items():
		match value:
			case Top():
				continue
			case Known(values=values):
				if program.machine is Machine.EM_386:
					stack[program.pointer_size * (position + 1)] = values
				elif program.machine is Machine.EM_X86_64:
					registers[X86_64_ARGUMENT_REGISTERS[position]] = values
				else:
					registers[ARM_ARGUMENT_REGISTERS[position]] = values
			case _ as unreachable:
				assert_never(unreachable)
	return State(
		registers=registers,
		sp_offsets={SP_REGISTERS[program.machine][0]: frozenset({0})},
		stack=stack,
		globals={},
	)
