# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Extraction of indirect call sites from machine code and per-site resolution.

Indirect call and tail-branch instructions are detected in each function's
bytes with capstone, then a ten-instruction window of register tracking maps
the target register back to a pointer value. Memory loads dereference
through the loaded image, so chains like ``mov rax, [rip+dev]`` /
``mov rax, [rax]`` resolve to the stored function pointer rather than
stopping at the first slot. Unconditional control transfers clear the
tracked values (registers are dead across them); conditional branches keep
them, since both paths to the site run the tracked instructions.

The linear sweep decodes past embedded data (jump tables, literal pools)
on architectures where every byte is a valid instruction; sites found in
such regions resolve to no candidates and fall back to the union of all
resolved targets, keeping the expansion a sound upper bound.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

from capstone import (
	CS_ARCH_ARM,
	CS_ARCH_X86,
	CS_MODE_32,
	CS_MODE_64,
	CS_MODE_THUMB,
	Cs,
	arm_const,
	x86_const,
)

from dynamic_call_tree_resolution.model import Address, CallSite
from dynamic_call_tree_resolution.points_to import memory_at, pointer_at

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping

	from capstone import CsInsn

	from dynamic_call_tree_resolution.model import Function, Program, SlotAssignment


class _MemoryOperand(Protocol):
	@property
	def base(self) -> int: ...  # pragma: no cover
	@property
	def index(self) -> int: ...  # pragma: no cover
	@property
	def disp(self) -> int: ...  # pragma: no cover


class _Operand(Protocol):
	@property
	def type(self) -> int: ...  # pragma: no cover
	@property
	def reg(self) -> int: ...  # pragma: no cover
	@property
	def imm(self) -> int: ...  # pragma: no cover
	@property
	def mem(self) -> _MemoryOperand | None: ...  # pragma: no cover


_WINDOW = 10

_DISASSEMBLERS: Mapping[str, tuple[int, int]] = {
	"EM_X86_64": (CS_ARCH_X86, CS_MODE_64),
	"EM_386": (CS_ARCH_X86, CS_MODE_32),
	"EM_ARM": (CS_ARCH_ARM, CS_MODE_THUMB),
}

_X86_TRANSFERS = (
	"call",
	"jmp",
	"ret",
	"retf",
	"iret",
	"loop",
	"loope",
	"loopne",
	"int",
	"int3",
	"syscall",
	"ud2",
	"hlt",
)
_ARM_TRANSFERS = ("bl", "blx", "bx", "b", "pop", "svc", "bkpt", "udf", "tbb", "tbh")
_X86_MOVES = ("mov", "movabs")
_ARM_LOADS = ("ldr", "ldr.w", "ldr.n")
_ARM_MOVES = ("mov", "movs", "mov.w")


def extract_call_sites(program: Program) -> tuple[CallSite, ...]:
	"""Extract every indirect call and tail-branch site from the program's code.

	Sites carry the address their target is taken from: the target function
	itself when the instruction's operand resolves to one, the slot holding
	the function pointer, or ``None`` when register tracking could not
	establish either. Unsupported machine types yield no sites.
	"""
	mode = _DISASSEMBLERS.get(program.machine)
	if mode is None:
		return ()
	disassembler = Cs(*mode)
	disassembler.detail = True
	disassembler.skipdata = True
	seen: set[Address] = set()
	sites: list[CallSite] = []
	for function in sorted(program.functions.values(), key=lambda function: function.address):
		# ARM symbol addresses carry the Thumb bit; strip it so the code
		# decodes from the aligned start, and skip the symbol/DWARF twin.
		start = Address(function.address & ~1) if program.machine == "EM_ARM" else function.address
		if start in seen:
			continue
		seen.add(start)
		sites.extend(_sites_in(disassembler, function, program, start))
	return tuple(sites)


def _sites_in(
	disassembler: Cs, function: Function, program: Program, start: Address
) -> Iterable[CallSite]:
	code = memory_at(program, start, function.size)
	if not code:
		return
	instructions = tuple(disassembler.disasm(code, start))
	for index, instruction in enumerate(instructions):
		site_operand = _site_operand(instruction, program.machine)
		if site_operand is None:
			continue
		kind, operand = site_operand
		state = _window_state(instructions, index, program)
		yield CallSite(
			caller_address=function.address,
			site_address=Address(instruction.address),
			slot=(
				_x86_memory_address(instruction, operand, state)
				if kind == "memory"
				else state.get(operand.reg)
			),
		)


def _site_operand(
	instruction: CsInsn, machine: str
) -> tuple[Literal["memory", "register"], _Operand] | None:
	if instruction.mnemonic == ".byte":
		return None
	if not instruction.operands:
		return None
	operand = instruction.operands[0]
	mnemonic = instruction.mnemonic
	if machine in ("EM_X86_64", "EM_386"):
		if mnemonic in ("call", "jmp"):
			if operand.type == x86_const.X86_OP_MEM:
				return "memory", operand
			if operand.type == x86_const.X86_OP_REG:
				return "register", operand
		return None
	if mnemonic == "blx" and operand.type == arm_const.ARM_OP_REG:
		return "register", operand
	if mnemonic == "bx" and operand.reg not in (arm_const.ARM_REG_LR, arm_const.ARM_REG_PC):
		return "register", operand
	return None


def _window_state(
	instructions: tuple[CsInsn, ...], site_index: int, program: Program
) -> dict[int, Address | None]:
	state: dict[int, Address | None] = {}
	for instruction in instructions[max(0, site_index - _WINDOW) : site_index]:
		if _is_control_transfer(instruction, program.machine):
			state = {}
			continue
		_apply(instruction, state, program)
	return state


def _is_control_transfer(instruction: CsInsn, machine: str) -> bool:
	if machine in ("EM_X86_64", "EM_386"):
		return instruction.mnemonic in _X86_TRANSFERS
	return instruction.mnemonic in _ARM_TRANSFERS


def _apply(instruction: CsInsn, state: dict[int, Address | None], program: Program) -> None:
	if instruction.mnemonic == ".byte":
		return
	if program.machine in ("EM_X86_64", "EM_386"):
		_apply_x86(instruction, state, program)
	else:
		_apply_arm(instruction, state, program)


def _apply_x86(instruction: CsInsn, state: dict[int, Address | None], program: Program) -> None:
	if instruction.mnemonic in _X86_MOVES:
		destination, source = instruction.operands
		if destination.type == x86_const.X86_OP_REG:
			match source.type:
				case x86_const.X86_OP_REG:
					state[destination.reg] = state.get(source.reg)
				case x86_const.X86_OP_IMM:
					state[destination.reg] = Address(source.imm)
				case x86_const.X86_OP_MEM:
					state[destination.reg] = _deref(
						_x86_memory_address(instruction, source, state), program
					)
				case _:
					pass
			return
	if instruction.mnemonic == "lea":
		destination, source = instruction.operands
		state[destination.reg] = _x86_memory_address(instruction, source, state)
		return
	for register in instruction.regs_access()[1]:
		state[register] = None


def _apply_arm(instruction: CsInsn, state: dict[int, Address | None], program: Program) -> None:
	if instruction.mnemonic in _ARM_LOADS:
		destination = instruction.operands[0]
		state[destination.reg] = _deref(
			_arm_memory_address(instruction, instruction.operands[1], state), program
		)
		return
	if instruction.mnemonic in _ARM_MOVES:
		destination = instruction.operands[0]
		source = instruction.operands[1]
		match source.type:
			case arm_const.ARM_OP_REG:
				state[destination.reg] = state.get(source.reg)
			case arm_const.ARM_OP_IMM:
				state[destination.reg] = Address(source.imm)
			case _:
				pass
		return
	for register in instruction.regs_access()[1]:
		state[register] = None


def _deref(address: Address | None, program: Program) -> Address | None:
	return None if address is None else pointer_at(program, address)


def _x86_memory_address(
	instruction: CsInsn,
	operand: _Operand,
	state: dict[int, Address | None],
) -> Address | None:
	memory = operand.mem
	if memory is None:
		return None  # pragma: no cover
	if memory.base == x86_const.X86_REG_RIP:
		return Address(instruction.address + instruction.size + memory.disp)
	if memory.base == 0:
		return Address(memory.disp)
	if memory.base in (x86_const.X86_REG_RSP, x86_const.X86_REG_RBP) or memory.index != 0:
		return None
	base = state.get(memory.base)
	return None if base is None else Address(base + memory.disp)


def _arm_memory_address(
	instruction: CsInsn,
	operand: _Operand,
	state: dict[int, Address | None],
) -> Address | None:
	memory = operand.mem
	if memory is None:
		return None  # pragma: no cover
	if memory.base == arm_const.ARM_REG_PC:
		return Address(((instruction.address + 4) & ~3) + memory.disp)
	if memory.base == arm_const.ARM_REG_SP or memory.index != 0:
		return None
	base = state.get(memory.base)
	return None if base is None else Address(base + memory.disp)


def call_site_candidates(
	program: Program, site: CallSite, resolved_by_slot: Mapping[Address, SlotAssignment]
) -> frozenset[Address]:
	"""Candidate target functions of one call site.

	An empty result means the site could not be resolved; consumers fall
	back to the union of all resolved targets.
	"""
	return _chase_target(program, site.slot, resolved_by_slot, frozenset())


def _chase_target(
	program: Program,
	address: Address | None,
	resolved_by_slot: Mapping[Address, SlotAssignment],
	visited: frozenset[Address],
) -> frozenset[Address]:
	if address is None or address in visited:
		return frozenset()
	if address in program.functions:
		return frozenset({address})
	assignment = resolved_by_slot.get(address)
	if assignment is not None:
		return assignment.candidates
	return _chase_target(
		program, pointer_at(program, address), resolved_by_slot, visited | {address}
	)


def per_caller_candidates(
	program: Program,
	sites: tuple[CallSite, ...],
	resolved: tuple[SlotAssignment, ...],
) -> tuple[Mapping[str, frozenset[str]], frozenset[str]]:
	"""Union, per caller, of each site's candidate target names, plus the fallback.

	Unresolved sites contribute the fallback union of all resolved targets,
	so a caller with an unresolved site is indistinguishable from a caller
	with no extracted sites; both keep the expansion a sound upper bound.
	"""
	resolved_by_slot = {assignment.slot: assignment for assignment in resolved}
	fallback_addresses = frozenset(
		address for assignment in resolved for address in assignment.candidates
	)
	by_caller: dict[str, set[str]] = {}
	for site in sites:
		caller = program.functions[site.caller_address].name
		candidates = call_site_candidates(program, site, resolved_by_slot) or fallback_addresses
		by_caller.setdefault(caller, set()).update(
			program.functions[address].name for address in candidates
		)
	return (
		{caller: frozenset(targets) for caller, targets in by_caller.items()},
		frozenset(program.functions[address].name for address in fallback_addresses),
	)
