# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Which writable data objects only the code that names them can reach (C's pointer provenance)."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from enum import Enum
from itertools import accumulate, chain
from typing import TYPE_CHECKING, Final, assert_never, cast

from capstone import CS_AC_READ, CS_AC_WRITE, arm_const
from salix import Struct

from dynamic_call_tree_resolution.model import Address, InstructionFamily
from dynamic_call_tree_resolution.points_to import in_writable_memory, instruction_set_at, memory_at
from dynamic_call_tree_resolution.vsa.abi import (
	ARM_ADDITION_SIGNS,
	ARM_ARGUMENT_REGISTERS,
	ARM_CALLER_SAVED,
	ARM_MULTIPLE_STORES,
	ARM_REGISTER_BYTES,
	ARM_RETURNING_TRAPS,
	ARM_STORE_WIDTHS,
	ARM_UNTRACKED_STORES,
	arm_predicated,
	is_data,
	normalized,
	program_counter,
)
from dynamic_call_tree_resolution.vsa.cfg import branch_target, call_target

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping

	from capstone import ArmCsOperand, CsInsn

	from dynamic_call_tree_resolution.model import Function, Program
	from dynamic_call_tree_resolution.vsa.cfg import Block

type Span = tuple[Address, Address]

_TRACKED_LIMIT: Final = 64
_NEARBY: Final = 0x1000
_WORD_MASK: Final = 0xFFFF_FFFF
_COMPARES: Final = frozenset(
	{
		arm_const.ARM_INS_CMP,
		arm_const.ARM_INS_CMN,
		arm_const.ARM_INS_TST,
		arm_const.ARM_INS_TEQ,
		arm_const.ARM_INS_CBZ,
		arm_const.ARM_INS_CBNZ,
	}
)
_MOVES: Final = frozenset({arm_const.ARM_INS_MOV, arm_const.ARM_INS_MOVS})
_DERIVATIONS: Final = _MOVES | {arm_const.ARM_INS_MOVT} | ARM_ADDITION_SIGNS.keys()
_LISTED_BASES: Final = frozenset(
	{
		arm_const.ARM_INS_LDM,
		arm_const.ARM_INS_LDMDA,
		arm_const.ARM_INS_LDMDB,
		arm_const.ARM_INS_LDMIB,
		arm_const.ARM_INS_STM,
		arm_const.ARM_INS_STMDA,
		arm_const.ARM_INS_STMDB,
		arm_const.ARM_INS_STMIB,
		arm_const.ARM_INS_VLDMIA,
		arm_const.ARM_INS_VLDMDB,
		arm_const.ARM_INS_VSTMIA,
		arm_const.ARM_INS_VSTMDB,
	}
)
_DESCENDING: Final = frozenset(
	{
		arm_const.ARM_INS_LDMDA,
		arm_const.ARM_INS_LDMDB,
		arm_const.ARM_INS_STMDA,
		arm_const.ARM_INS_STMDB,
		arm_const.ARM_INS_VLDMDB,
		arm_const.ARM_INS_VSTMDB,
	}
)
_PLACED_STORES: Final = (frozenset(ARM_STORE_WIDTHS) | ARM_MULTIPLE_STORES) - {
	arm_const.ARM_INS_PUSH,
	arm_const.ARM_INS_VPUSH,
}
_DISPATCHES: Final = frozenset({arm_const.ARM_INS_TBB, arm_const.ARM_INS_TBH})
_RETURNED: Final = frozenset({arm_const.ARM_REG_R0, arm_const.ARM_REG_R1})


class Escapes(Struct):
	"""What the code lets escape, which bounds what a store the analysis can't place may write."""

	unescaped: tuple[Span, ...]
	"""Writable objects whose address no escaped value can reach, sorted."""
	store_targets: Mapping[Address, frozenset[Span]]
	"""The unescaped objects each store, by address, may write through an address the code derived
	from them."""


class _Derived(Struct):
	"""An address the code computed from one it materialized."""

	origin: int
	address: int
	site: int
	"""The instruction that materialized the origin."""


class _Held(Struct):
	"""The addresses one register may hold."""

	exact: frozenset[_Derived] = frozenset()
	indexed: frozenset[_Derived] = frozenset()
	"""Addresses with an offset the scan doesn't know added."""


_NOTHING: Final = _Held()


class _Registers(Struct):
	held: Mapping[int, _Held]
	constants: Mapping[int, int]
	"""What every path into the instruction leaves in the register."""


_UNKNOWN: Final = _Registers(held={}, constants={})


class _Fill(Enum):
	COPY = "copy"
	ZERO = "zero"


_FILL_ROUTINES: Final = {"memcpy": _Fill.COPY, "memset": _Fill.ZERO}


class _Call(Struct):
	target: Address | None


class _Return(Struct):
	passed: frozenset[int]
	"""The registers that carry the return value."""


class _Jump(Struct):
	"""A write to the program counter that neither calls nor returns."""

	resolved: bool
	"""The control flow graph knows where it goes."""


type _Control = _Call | _Return | _Jump


class _Scope(Struct):
	"""What the scan of one function needs to know about it."""

	blocks: frozenset[Address]
	returned: frozenset[int]
	"""The registers its return value occupies; none when DWARF says it returns ``void``."""


class _Image(Struct):
	program: Program
	spans: tuple[Span, ...]
	"""Writable objects, with overlapping objects merged, sorted."""
	starts: tuple[Address, ...]
	nearby: tuple[int, int]
	"""The addresses close enough to a writable object to follow as possible addresses."""
	bounded: Mapping[int, tuple[Span, ...]]
	"""The writable sections each linker label bounds."""
	walkable: Mapping[int, tuple[Span, ...]]
	"""Of those, the ones code may walk as an array from the label: every one at a label that
	starts no object, else those whose objects share one type, as an iterable section's do."""
	fills: Mapping[Address, _Fill]
	"""``memcpy``, ``memset``, and functions whose only instruction branches to one of them."""
	parameterless: frozenset[Address]
	"""Functions DWARF defines with no parameters, which read no argument registers."""


class _Effects(Struct):
	escaped: _Held = _NOTHING
	filled: _Held = _NOTHING
	"""The destination of a copy or zero fill whose arguments aren't constant."""
	overwritten: frozenset[Span] = frozenset()
	"""Objects an initialization call leaves without their image value."""
	site: Address | None = None
	"""The store instruction's address."""
	store: _Held = _NOTHING
	"""Where the store writes."""
	literal: int | None = None
	"""The address of the word a PC-relative load reads."""


_NO_EFFECTS: Final = _Effects()


class _Scan(Struct):
	escaped: frozenset[Span]
	stores: Mapping[Address, frozenset[Span]]
	literals: frozenset[int]


def escapes(program: Program, graphs: tuple[tuple[Function, tuple[Block, ...]], ...]) -> Escapes:
	match program.machine.family:
		case InstructionFamily.X86:
			return Escapes(unescaped=(), store_targets={})
		case InstructionFamily.ARM:
			return _arm_escapes(_image(program, graphs), graphs)
		case _ as unreachable:
			assert_never(unreachable)


def _arm_escapes(image: _Image, graphs: tuple[tuple[Function, tuple[Block, ...]], ...]) -> Escapes:
	scans = tuple(_scan(image, function, blocks) for function, blocks in graphs)
	escaped = frozenset(chain.from_iterable(scan.escaped for scan in scans)) | _data_escapes(
		image, frozenset(chain.from_iterable(scan.literals for scan in scans))
	)
	return Escapes(
		unescaped=tuple(span for span in image.spans if span not in escaped),
		store_targets={
			site: targets
			for scan in scans
			for site, reached in scan.stores.items()
			if (targets := reached - escaped)
		},
	)


def _image(program: Program, graphs: tuple[tuple[Function, tuple[Block, ...]], ...]) -> _Image:
	spans = _merged(
		(data_object.address, Address(data_object.address + data_object.size))
		for data_object in program.objects.values()
		if data_object.size > 0 and in_writable_memory(program, data_object.address)
	)
	writable = tuple(
		sorted(
			(
				*(
					(start, Address(start + len(section.data)))
					for start, section in program.sections.items()
					if section.writable and section.data
				),
				*program.nobits,
			)
		)
	)
	bounds = (*spans, *writable)
	bounded: Mapping[int, tuple[Span, ...]] = {
		address: sections
		for address in program.labels.values()
		if (sections := tuple(span for span in writable if span[0] <= address <= span[1]))
	}
	object_starts = frozenset(data_object.address for data_object in program.objects.values())
	arrays = frozenset(section for section in writable if _one_type(program, section))
	routines = {
		normalized(address, program.machine): fill
		for name, fill in _FILL_ROUTINES.items()
		for address in program.symbol_addresses.get(name, frozenset())
	}
	return _Image(
		program=program,
		spans=spans,
		starts=tuple(start for start, _ in spans),
		nearby=(
			min(low for low, _ in bounds) - _NEARBY,
			max(high for _, high in bounds) + _NEARBY,
		)
		if bounds
		else (0, -1),
		bounded=bounded,
		walkable={
			address: walkable
			for address, sections in bounded.items()
			if (
				walkable := tuple(
					section
					for section in sections
					if address not in object_starts or section in arrays
				)
			)
		},
		parameterless=frozenset(
			normalized(function.address, program.machine)
			for function in program.functions.values()
			if function.signature is not None and not function.signature.parameters
		),
		fills={
			**routines,
			**{
				normalized(function.address, program.machine): fill
				for function, blocks in graphs
				if (fill := _forwarded_fill(program, routines, blocks)) is not None
			},
		},
	)


def _one_type(program: Program, section: Span) -> bool:
	return (
		len(
			{
				data_object.type_name
				for data_object in program.objects.values()
				if data_object.type_name is not None
				and section[0] <= data_object.address < section[1]
			}
		)
		<= 1
	)


def _merged(spans: Iterable[Span]) -> tuple[Span, ...]:
	ordered = sorted(spans)
	reach = tuple(accumulate((end for _, end in ordered), max))
	breaks = tuple(
		index
		for index in range(len(ordered))
		if index == 0 or ordered[index][0] >= reach[index - 1]
	)
	return tuple(
		(ordered[first][0], reach[after - 1])
		for first, after in zip(breaks, (*breaks[1:], len(ordered)), strict=True)
	)


def _forwarded_fill(
	program: Program, routines: Mapping[Address, _Fill], blocks: tuple[Block, ...]
) -> _Fill | None:
	match tuple(
		instruction
		for block in blocks
		for instruction in block.instructions
		if not is_data(instruction)
	):
		case (only,):
			branch = branch_target(only, program.machine)
			return (
				routines.get(normalized(Address(branch.target), program.machine))
				if branch is not None and not branch.conditional
				else None
			)
		case _:
			return None


def _scan(image: _Image, function: Function, blocks: tuple[Block, ...]) -> _Scan:
	by_start = {block.start: block for block in blocks}
	scope = _Scope(
		blocks=frozenset(by_start),
		returned=frozenset()
		if function.signature is not None and function.signature.return_type == "void"
		else _RETURNED,
	)
	in_states: dict[Address, _Registers] = {}
	effects: list[_Effects] = []
	for seed in by_start:
		if seed in in_states:
			continue
		in_states[seed] = _UNKNOWN
		worklist = [seed]
		while worklist:
			block = by_start[worklist.pop()]
			registers = in_states[block.start]
			for instruction in block.instructions:
				registers, effect = _step(image, scope, instruction, registers, block.successors)
				effects.append(effect)
			for successor in block.successors:
				joined, overflow = _join(
					in_states.get(successor), registers, back=successor <= block.start
				)
				if _size(overflow):
					effects.append(_Effects(escaped=overflow))
				if joined != in_states.get(successor):
					in_states[successor] = joined
					worklist.append(successor)
	return _scanned(image, tuple(effects))


def _scanned(image: _Image, effects: tuple[_Effects, ...]) -> _Scan:
	stores: defaultdict[Address, set[Span]] = defaultdict(set)
	for effect in effects:
		if effect.site is not None:
			stores[effect.site].update(_reached(image, effect.store))
	return _Scan(
		escaped=frozenset(
			chain(
				chain.from_iterable(_reached(image, effect.escaped) for effect in effects),
				chain.from_iterable(
					_containing(image, derived.address)
					| _sections_objects(image, image.bounded, derived.address)
					| _sections_objects(image, image.bounded, derived.origin)
					for effect in effects
					for derived in chain(effect.filled.exact, effect.filled.indexed)
				),
				chain.from_iterable(effect.overwritten for effect in effects),
			)
		),
		stores={site: frozenset(targets) for site, targets in stores.items() if targets},
		literals=frozenset(effect.literal for effect in effects if effect.literal is not None),
	)


def _join(
	current: _Registers | None, incoming: _Registers, *, back: bool
) -> tuple[_Registers, _Held]:
	held, overflow = _capped(
		incoming.held
		if current is None
		else {
			register: (_carried if back else _union)(
				current.held.get(register, _NOTHING), incoming.held.get(register, _NOTHING)
			)
			for register in current.held.keys() | incoming.held.keys()
		}
	)
	return _Registers(
		held=held,
		constants=incoming.constants
		if current is None
		else {
			register: value
			for register, value in current.constants.items()
			if incoming.constants.get(register) == value
		},
	), overflow


def _carried(current: _Held, incoming: _Held) -> _Held:
	"""The join along an edge back into a loop.

	An address the loop moves stays in the object it entered the loop pointing into (C's pointer
	provenance), so it becomes that address plus an unknown offset.
	"""
	moved = frozenset(
		_lineage(derived)
		for derived in chain(incoming.exact, incoming.indexed)
		if derived not in current.exact and derived not in current.indexed
	) & frozenset(_lineage(derived) for derived in chain(current.exact, current.indexed))
	return _Held(
		exact=frozenset(
			derived for derived in current.exact | incoming.exact if _lineage(derived) not in moved
		),
		indexed=frozenset(
			derived
			for derived in chain(current.indexed, incoming.indexed)
			if _lineage(derived) not in moved
		)
		| frozenset(
			derived
			for derived in chain(current.exact, current.indexed)
			if _lineage(derived) in moved
		),
	)


def _lineage(derived: _Derived) -> tuple[int, int]:
	return derived.origin, derived.site


def _capped(held: Mapping[int, _Held]) -> tuple[Mapping[int, _Held], _Held]:
	"""The registers that hold few enough addresses to follow, and what the rest held."""
	return (
		{register: value for register, value in held.items() if 0 < _size(value) <= _TRACKED_LIMIT},
		_union(*(value for value in held.values() if _size(value) > _TRACKED_LIMIT)),
	)


def _size(held: _Held) -> int:
	return len(held.exact) + len(held.indexed)


def _union(*held: _Held) -> _Held:
	match tuple(value for value in held if value.exact or value.indexed):
		case ():
			return _NOTHING
		case (only,):
			return only
		case several:
			return _Held(
				exact=frozenset[_Derived]().union(*(value.exact for value in several)),
				indexed=frozenset[_Derived]().union(*(value.indexed for value in several)),
			)


def _offset(held: _Held, delta: int) -> _Held:
	return _Held(
		exact=frozenset(_moved_by(derived, delta) for derived in held.exact),
		indexed=frozenset(_moved_by(derived, delta) for derived in held.indexed),
	)


def _moved_by(derived: _Derived, delta: int) -> _Derived:
	return _Derived(
		origin=derived.origin, address=(derived.address + delta) & _WORD_MASK, site=derived.site
	)


def _unknown_offset(held: _Held) -> _Held:
	return _Held(indexed=held.exact | held.indexed)


def _step(
	image: _Image,
	scope: _Scope,
	instruction: CsInsn,
	registers: _Registers,
	successors: tuple[Address, ...],
) -> tuple[_Registers, _Effects]:
	if is_data(instruction) or instruction.id == arm_const.ARM_INS_IT:
		return registers, _NO_EFFECTS
	operands = _operands(instruction)
	written = frozenset(instruction.regs_access()[1])
	control = _control(image, scope, instruction, operands, written, successors)
	fill = _filled(image, control, registers.constants)
	after = _after(
		instruction,
		registers,
		{
			**_destination(image, instruction, operands, registers),
			**_written_back(instruction, operands, registers),
		},
		written | (frozenset(ARM_CALLER_SAVED) if isinstance(control, _Call) else frozenset[int]()),
	)
	return after, _Effects(
		escaped=_union(
			*(
				registers.held.get(register, _NOTHING)
				for register in _escaping(image, instruction, operands, control, fill, registers)
			)
		),
		filled=_walked(image, control, fill, registers),
		overwritten=frozenset() if fill is None else fill,
		site=Address(instruction.address) if instruction.id in _PLACED_STORES else None,
		store=_store_target(instruction, operands, registers),
		literal=_literal_address(image.program, instruction, operands),
	)


def _after(
	instruction: CsInsn,
	registers: _Registers,
	new: Mapping[int, tuple[_Held, int | None]],
	written: frozenset[int],
) -> _Registers:
	"""The registers after the instruction; a predicated one may not run, so it only adds."""
	predicated = arm_predicated(instruction)
	kept = {
		register: held
		for register, held in registers.held.items()
		if predicated or register not in written
	}
	return _Registers(
		held={
			**kept,
			**{
				register: value
				for register, value in (
					(register, _union(kept.get(register, _NOTHING), value) if predicated else value)
					for register, (value, _) in new.items()
				)
				if _size(value)
			},
		},
		constants={
			**{
				register: value
				for register, value in registers.constants.items()
				if register not in written
				or (predicated and register in new and new[register][1] == value)
			},
			**(
				{}
				if predicated
				else {register: value for register, (_, value) in new.items() if value is not None}
			),
		},
	)


def _control(
	image: _Image,
	scope: _Scope,
	instruction: CsInsn,
	operands: tuple[ArmCsOperand, ...],
	written: frozenset[int],
	successors: tuple[Address, ...],
) -> _Control | None:
	machine = image.program.machine
	called = call_target(instruction, machine)
	if called is not None:
		return _Call(target=normalized(Address(called), machine))
	if instruction.id in ARM_RETURNING_TRAPS:
		return _Call(target=None)
	branch = branch_target(instruction, machine)
	if branch is not None:
		return (
			None
			if normalized(Address(branch.target), machine) in scope.blocks
			else _Call(target=normalized(Address(branch.target), machine))
		)
	if instruction.id in (arm_const.ARM_INS_BX, arm_const.ARM_INS_BLX):
		return (
			_Return(passed=scope.returned)
			if instruction.id == arm_const.ARM_INS_BX and operands[0].reg == arm_const.ARM_REG_LR
			else _Call(target=None)
		)
	if instruction.id in _DISPATCHES or arm_const.ARM_REG_PC in written:
		return (
			_Return(passed=scope.returned)
			if _returns(instruction, operands)
			else _Jump(resolved=len(successors) > (1 if arm_predicated(instruction) else 0))
		)
	return None


def _returns(instruction: CsInsn, operands: tuple[ArmCsOperand, ...]) -> bool:
	"""Whether a write to the program counter takes it from the stack or the link register."""
	return (
		instruction.id == arm_const.ARM_INS_POP
		or any(
			operand.type == arm_const.ARM_OP_MEM and operand.mem.base == arm_const.ARM_REG_SP
			for operand in operands
		)
		or (instruction.id in _LISTED_BASES and operands[0].reg == arm_const.ARM_REG_SP)
		or (
			instruction.id in _MOVES
			and operands[1].type == arm_const.ARM_OP_REG
			and operands[1].reg == arm_const.ARM_REG_LR
		)
	)


def _filled(
	image: _Image, control: _Control | None, constants: Mapping[int, int]
) -> frozenset[Span] | None:
	"""The objects a fill with constant arguments leaves without their image value, if it is one."""
	match (
		image.fills.get(control.target)
		if isinstance(control, _Call) and control.target is not None
		else None,
		constants.get(arm_const.ARM_REG_R0),
		constants.get(arm_const.ARM_REG_R1),
		constants.get(arm_const.ARM_REG_R2),
	):
		case _Fill() as fill, int() as destination, int() as argument, int() as length:
			return frozenset(
				span
				for span in _overlapping(image, destination, destination + length)
				if not _reproduced(image, fill, span, destination, argument)
			)
		case _:
			return None


def _walked(
	image: _Image, control: _Control | None, fill: frozenset[Span] | None, registers: _Registers
) -> _Held:
	"""The destination of a fill whose arguments aren't constant, which it writes as a range."""
	return (
		_unknown_offset(registers.held.get(ARM_ARGUMENT_REGISTERS[0], _NOTHING))
		if fill is None
		and isinstance(control, _Call)
		and control.target is not None
		and control.target in image.fills
		else _NOTHING
	)


def _reproduced(image: _Image, fill: _Fill, span: Span, destination: int, argument: int) -> bool:
	match fill:
		case _Fill.COPY:
			return any(
				section.writable
				and section.load_address is not None
				and start <= span[0]
				and span[1] <= start + len(section.data)
				and section.load_address - start == argument - destination
				for start, section in image.program.sections.items()
			)
		case _Fill.ZERO:
			return argument == 0 and any(
				low <= span[0] and span[1] <= high for low, high in image.program.nobits
			)
		case _ as unreachable:
			assert_never(unreachable)


def _escaping(
	image: _Image,
	instruction: CsInsn,
	operands: tuple[ArmCsOperand, ...],
	control: _Control | None,
	fill: frozenset[Span] | None,
	registers: _Registers,
) -> frozenset[int]:
	"""The registers whose addresses the instruction puts somewhere the scan doesn't follow."""
	listed = frozenset({operands[0].reg}) if instruction.id in _LISTED_BASES else frozenset[int]()
	memory = tuple(operand.mem for operand in operands if operand.type == arm_const.ARM_OP_MEM)
	return (
		frozenset(access.index for access in memory if access.index)
		| (
			frozenset(access.base for access in memory) | listed
			if instruction.id in ARM_UNTRACKED_STORES
			else frozenset[int]()
		)
		| (
			frozenset(
				operand.reg
				for operand in operands
				if operand.type == arm_const.ARM_OP_REG and operand.access & CS_AC_READ
			)
			- _consumed(instruction, operands, listed)
		)
		| _passed(image, control, fill, registers)
	)


def _consumed(
	instruction: CsInsn, operands: tuple[ArmCsOperand, ...], listed: frozenset[int]
) -> frozenset[int]:
	"""The registers the instruction reads only to compare or to derive another address."""
	return (
		frozenset(operand.reg for operand in operands if operand.type == arm_const.ARM_OP_REG)
		if instruction.id in _COMPARES
		or (
			instruction.id in _DERIVATIONS
			and operands[0].reg not in (arm_const.ARM_REG_SP, arm_const.ARM_REG_PC)
		)
		else listed
	)


def _passed(
	image: _Image, control: _Control | None, fill: frozenset[Span] | None, registers: _Registers
) -> frozenset[int]:
	match control:
		case None:
			return frozenset()
		case _Call(target=target):
			return (
				frozenset()
				if fill is not None or target in image.parameterless
				else frozenset(ARM_ARGUMENT_REGISTERS)
			)
		case _Return(passed=passed):
			return passed
		case _Jump(resolved=resolved):
			return frozenset() if resolved else frozenset(registers.held)
		case _ as unreachable:
			assert_never(unreachable)


def _destination(
	image: _Image, instruction: CsInsn, operands: tuple[ArmCsOperand, ...], registers: _Registers
) -> Mapping[int, tuple[_Held, int | None]]:
	if (
		not operands
		or operands[0].type != arm_const.ARM_OP_REG
		or not operands[0].access & CS_AC_WRITE
	):
		return {}
	constant = _constant(image.program, instruction, operands, registers.constants)
	held = _destination_held(image, instruction, operands, registers, constant)
	return {operands[0].reg: (held, constant)} if constant is not None or _size(held) else {}


def _constant(
	program: Program,
	instruction: CsInsn,
	operands: tuple[ArmCsOperand, ...],
	constants: Mapping[int, int],
) -> int | None:
	"""The value every path gives the instruction's destination, when the scan can compute it."""
	match instruction.id, operands:
		case arm_const.ARM_INS_LDR, _ if (
			literal := _literal_address(program, instruction, operands)
		) is not None:
			return _word(program, literal)
		case arm_const.ARM_INS_ADR, (_, offset):
			return (_pc(program, instruction) + offset.imm) & _WORD_MASK
		case identifier, (_, base, offset) if (
			identifier in ARM_ADDITION_SIGNS
			and base.type == arm_const.ARM_OP_REG
			and base.reg == arm_const.ARM_REG_PC
			and offset.type == arm_const.ARM_OP_IMM
		):
			return (
				_pc(program, instruction) + ARM_ADDITION_SIGNS[identifier] * offset.imm
			) & _WORD_MASK
		case ((arm_const.ARM_INS_MOV | arm_const.ARM_INS_MOVS), (_, source)):
			return _masked(_operand_constant(source, constants))
		case arm_const.ARM_INS_MOVT, (destination, high) if destination.reg in constants:
			return ((high.imm << 16) | (constants[destination.reg] & 0xFFFF)) & _WORD_MASK
		case identifier, _ if identifier in ARM_ADDITION_SIGNS:
			match (
				constants.get(_terms(operands)[0].reg),
				_operand_constant(_terms(operands)[1], constants),
			):
				case int() as augend, int() as addend:
					return (augend + ARM_ADDITION_SIGNS[identifier] * addend) & _WORD_MASK
				case _:
					return None
		case _:
			return None


def _masked(value: int | None) -> int | None:
	return None if value is None else value & _WORD_MASK


def _word(program: Program, address: int) -> int | None:
	data = memory_at(program, Address(address), 4)
	return int.from_bytes(data, program.byte_order) if len(data) == 4 else None


def _destination_held(
	image: _Image,
	instruction: CsInsn,
	operands: tuple[ArmCsOperand, ...],
	registers: _Registers,
	constant: int | None,
) -> _Held:
	if instruction.id in _MOVES and operands[1].type == arm_const.ARM_OP_REG:
		return registers.held.get(operands[1].reg, _NOTHING)
	if instruction.id in ARM_ADDITION_SIGNS and _terms(operands)[0].reg != arm_const.ARM_REG_PC:
		return _sum(instruction, operands, registers)
	return (
		_Held(
			exact=frozenset({_Derived(origin=constant, address=constant, site=instruction.address)})
		)
		if constant is not None and image.nearby[0] <= constant <= image.nearby[1]
		else _NOTHING
	)


def _sum(instruction: CsInsn, operands: tuple[ArmCsOperand, ...], registers: _Registers) -> _Held:
	sign = ARM_ADDITION_SIGNS[instruction.id]
	left, right = _terms(operands)
	if right.type == arm_const.ARM_OP_IMM:
		return _offset(registers.held.get(left.reg, _NOTHING), sign * right.imm)
	return _union(
		_offset_or_unknown(
			registers.held.get(left.reg, _NOTHING),
			_contribution(registers.constants.get(right.reg), right),
			sign,
		),
		_NOTHING
		if sign < 0 and left.reg in registers.constants
		else _offset_or_unknown(
			registers.held.get(right.reg, _NOTHING),
			registers.constants.get(left.reg)
			if right.shift.type == arm_const.ARM_SFT_INVALID
			else None,
			sign,
		),
	)


def _offset_or_unknown(held: _Held, delta: int | None, sign: int) -> _Held:
	return _unknown_offset(held) if delta is None else _offset(held, sign * delta)


def _terms(operands: tuple[ArmCsOperand, ...]) -> tuple[ArmCsOperand, ArmCsOperand]:
	"""An addition's two terms; a two-operand form adds to its destination."""
	return (operands[1], operands[2]) if len(operands) == 3 else (operands[0], operands[1])


def _operand_constant(operand: ArmCsOperand, constants: Mapping[int, int]) -> int | None:
	return (
		operand.imm
		if operand.type == arm_const.ARM_OP_IMM
		else _contribution(constants.get(operand.reg), operand)
		if operand.type == arm_const.ARM_OP_REG
		else None
	)


def _contribution(value: int | None, operand: ArmCsOperand) -> int | None:
	"""What a register operand holding the value adds, after its shift and sign."""
	if value is None or operand.shift.type not in (
		arm_const.ARM_SFT_INVALID,
		arm_const.ARM_SFT_LSL,
	):
		return None
	return (-1 if operand.subtracted else 1) * (
		value << (operand.shift.value if operand.shift.type == arm_const.ARM_SFT_LSL else 0)
	)


def _written_back(
	instruction: CsInsn, operands: tuple[ArmCsOperand, ...], registers: _Registers
) -> Mapping[int, tuple[_Held, int | None]]:
	"""The base register a load or store moves past what it accessed."""
	if instruction.id in _LISTED_BASES:
		return (
			{
				operands[0].reg: _moved(
					registers,
					operands[0].reg,
					-_listed_width(instruction, operands)
					if instruction.id in _DESCENDING
					else _listed_width(instruction, operands),
				)
			}
			if instruction.writeback
			else {}
		)
	memory = next(
		(index for index, operand in enumerate(operands) if operand.type == arm_const.ARM_OP_MEM),
		None,
	)
	if memory is None:
		return {}
	if memory < len(operands) - 1:
		return {
			operands[memory].mem.base: _moved(
				registers,
				operands[memory].mem.base,
				_operand_constant(operands[-1], registers.constants),
			)
		}
	if instruction.writeback:
		return {
			operands[memory].mem.base: _moved(
				registers,
				operands[memory].mem.base,
				None if operands[memory].mem.index else operands[memory].mem.disp,
			)
		}
	return {}


def _moved(registers: _Registers, base: int, delta: int | None) -> tuple[_Held, int | None]:
	return (
		(_unknown_offset(registers.held.get(base, _NOTHING)), None)
		if delta is None
		else (
			_offset(registers.held.get(base, _NOTHING), delta),
			_masked(None if base not in registers.constants else registers.constants[base] + delta),
		)
	)


def _listed_width(instruction: CsInsn, operands: tuple[ArmCsOperand, ...]) -> int:
	return sum(
		ARM_REGISTER_BYTES.get(instruction.reg_name(operand.reg)[0], 4) for operand in operands[1:]
	)


def _store_target(
	instruction: CsInsn, operands: tuple[ArmCsOperand, ...], registers: _Registers
) -> _Held:
	"""Where a store the analysis can place writes, by what its base register holds."""
	if instruction.id not in _PLACED_STORES:
		return _NOTHING
	if instruction.id in _LISTED_BASES:
		return _unknown_offset(
			_offset(
				registers.held.get(operands[0].reg, _NOTHING),
				-_listed_width(instruction, operands) if instruction.id in _DESCENDING else 0,
			)
		)
	memory = next(operand.mem for operand in operands if operand.type == arm_const.ARM_OP_MEM)
	return (
		_unknown_offset(_offset(registers.held.get(memory.base, _NOTHING), memory.disp))
		if memory.index
		else _offset(registers.held.get(memory.base, _NOTHING), memory.disp)
	)


def _literal_address(
	program: Program, instruction: CsInsn, operands: tuple[ArmCsOperand, ...]
) -> int | None:
	"""The word a PC-relative ``ldr`` reads."""
	return (
		(_pc(program, instruction) + operands[1].mem.disp) & _WORD_MASK
		if instruction.id == arm_const.ARM_INS_LDR
		and len(operands) == 2
		and operands[1].type == arm_const.ARM_OP_MEM
		and operands[1].mem.base == arm_const.ARM_REG_PC
		and not operands[1].mem.index
		else None
	)


def _pc(program: Program, instruction: CsInsn) -> int:
	return program_counter(instruction, instruction_set_at(program, Address(instruction.address)))


def _operands(instruction: CsInsn) -> tuple[ArmCsOperand, ...]:
	return tuple(cast("ArmCsOperand", operand) for operand in instruction.operands)


def _reached(image: _Image, held: _Held) -> frozenset[Span]:
	"""The objects the held addresses point into, under C's pointer provenance."""
	return frozenset(
		chain(
			chain.from_iterable(_attributed(image, derived.address) for derived in held.exact),
			chain.from_iterable(
				_containing(image, derived.address)
				| _sections_objects(image, image.walkable, derived.address)
				| _sections_objects(image, image.walkable, derived.origin)
				for derived in held.indexed
			),
		)
	)


def _attributed(image: _Image, address: int) -> frozenset[Span]:
	"""The object an address points into, and the objects code may walk from a label there."""
	return _containing(image, address) | _sections_objects(image, image.walkable, address)


def _containing(image: _Image, address: int) -> frozenset[Span]:
	index = bisect_right(image.starts, address) - 1
	return (
		frozenset({image.spans[index]})
		if index >= 0 and address < image.spans[index][1]
		else frozenset()
	)


def _overlapping(image: _Image, low: int, high: int) -> frozenset[Span]:
	return frozenset(
		span
		for span in image.spans[
			max(bisect_right(image.starts, low) - 1, 0) : bisect_left(image.starts, high)
		]
		if span[1] > low
	)


def _sections_objects(
	image: _Image, sections: Mapping[int, tuple[Span, ...]], address: int
) -> frozenset[Span]:
	return frozenset(
		chain.from_iterable(
			_overlapping(image, low, high) for low, high in sections.get(address, ())
		)
	)


def _data_escapes(image: _Image, literals: frozenset[int]) -> frozenset[Span]:
	"""The objects a word of data at any byte offset points into, outside code and its literals."""
	program = image.program
	return frozenset(
		chain.from_iterable(
			_attributed(image, word)
			for address, word in chain(
				chain.from_iterable(
					_words(program, start, section.data)
					for start, section in program.sections.items()
					if not section.executable
				),
				chain.from_iterable(
					_words(program, low, memory_at(program, low, high - low))
					for low, high in program.data_in_code
				),
			)
			if image.nearby[0] <= word <= image.nearby[1] and address not in literals
		)
	)


def _words(program: Program, start: int, data: bytes) -> Iterable[tuple[int, int]]:
	"""Each word at each byte offset of the data that starts at the address, with its address."""
	return (
		(start + offset, int.from_bytes(data[offset : offset + 4], program.byte_order))
		for offset in range(len(data) - 3)
	)
