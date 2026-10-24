# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Per-machine registers and instruction classes the analysis models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, assert_never, cast

from capstone import (
	CS_ARCH_ARM,
	CS_ARCH_X86,
	CS_MODE_32,
	CS_MODE_64,
	CS_MODE_ARM,
	CS_MODE_MCLASS,
	CS_MODE_THUMB,
	Cs,
	arm_const,
	x86_const,
)
from salix import Struct

from dynamic_call_tree_resolution.model import (
	Address,
	ArmCore,
	ArmProfile,
	InstructionFamily,
	InstructionSet,
	Machine,
	aligned,
)

if TYPE_CHECKING:
	from collections.abc import Mapping

	from capstone import ArmCsOperand, CsInsn, CsOperand


class CapstoneMode(Struct):
	"""A capstone architecture, with the mode to decode it in."""

	architecture: int
	mode: int


DISASSEMBLERS: Final[Mapping[InstructionSet, CapstoneMode]] = {
	InstructionSet.X86_64: CapstoneMode(architecture=CS_ARCH_X86, mode=CS_MODE_64),
	InstructionSet.X86_32: CapstoneMode(architecture=CS_ARCH_X86, mode=CS_MODE_32),
	InstructionSet.A32: CapstoneMode(architecture=CS_ARCH_ARM, mode=CS_MODE_ARM),
	InstructionSet.T32: CapstoneMode(architecture=CS_ARCH_ARM, mode=CS_MODE_THUMB),
}


def _stack_pointers(machine: Machine) -> tuple[int, ...]:
	match machine:
		case Machine.EM_X86_64:
			return (x86_const.X86_REG_RSP, x86_const.X86_REG_RBP)
		case Machine.EM_386:
			return (x86_const.X86_REG_ESP, x86_const.X86_REG_EBP)
		case Machine.EM_ARM:
			return (arm_const.ARM_REG_SP,)
		case _ as unreachable:
			assert_never(unreachable)


SP_REGISTERS: Final = {machine: _stack_pointers(machine) for machine in Machine}

X86_TRANSFERS: Final = (
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
ARM_TRANSFERS: Final = frozenset(
	{
		arm_const.ARM_INS_BL,
		arm_const.ARM_INS_BLX,
		arm_const.ARM_INS_BX,
		arm_const.ARM_INS_B,
		arm_const.ARM_INS_POP,
		arm_const.ARM_INS_SVC,
		arm_const.ARM_INS_BKPT,
		arm_const.ARM_INS_UDF,
		arm_const.ARM_INS_TBB,
		arm_const.ARM_INS_TBH,
	}
)
X86_CALLS: Final = ("call",)
X86_RETURNING_TRAPS: Final = ("syscall", "int", "int3", "hlt")
ARM_RETURNING_TRAPS: Final = frozenset({arm_const.ARM_INS_SVC, arm_const.ARM_INS_BKPT})
ARM_CALLS: Final = frozenset({arm_const.ARM_INS_BL, arm_const.ARM_INS_BLX})


class X86Register(Struct):
	"""Where an x86 general-purpose register sits in its family."""

	wide: int
	"""The family's 64-bit register."""
	narrow: int
	"""The family's 32-bit register."""
	bits: int


class _X86Family(Struct):
	"""One x86 general-purpose register at each of its widths."""

	wide: int
	narrow: int
	word: int
	bytes: tuple[int, ...]


_X86_FAMILIES: Final = (
	_X86Family(
		wide=x86_const.X86_REG_RAX,
		narrow=x86_const.X86_REG_EAX,
		word=x86_const.X86_REG_AX,
		bytes=(
			x86_const.X86_REG_AL,
			x86_const.X86_REG_AH,
		),
	),
	_X86Family(
		wide=x86_const.X86_REG_RBX,
		narrow=x86_const.X86_REG_EBX,
		word=x86_const.X86_REG_BX,
		bytes=(
			x86_const.X86_REG_BL,
			x86_const.X86_REG_BH,
		),
	),
	_X86Family(
		wide=x86_const.X86_REG_RCX,
		narrow=x86_const.X86_REG_ECX,
		word=x86_const.X86_REG_CX,
		bytes=(
			x86_const.X86_REG_CL,
			x86_const.X86_REG_CH,
		),
	),
	_X86Family(
		wide=x86_const.X86_REG_RDX,
		narrow=x86_const.X86_REG_EDX,
		word=x86_const.X86_REG_DX,
		bytes=(
			x86_const.X86_REG_DL,
			x86_const.X86_REG_DH,
		),
	),
	_X86Family(
		wide=x86_const.X86_REG_RSI,
		narrow=x86_const.X86_REG_ESI,
		word=x86_const.X86_REG_SI,
		bytes=(x86_const.X86_REG_SIL,),
	),
	_X86Family(
		wide=x86_const.X86_REG_RDI,
		narrow=x86_const.X86_REG_EDI,
		word=x86_const.X86_REG_DI,
		bytes=(x86_const.X86_REG_DIL,),
	),
	_X86Family(
		wide=x86_const.X86_REG_RBP,
		narrow=x86_const.X86_REG_EBP,
		word=x86_const.X86_REG_BP,
		bytes=(x86_const.X86_REG_BPL,),
	),
	_X86Family(
		wide=x86_const.X86_REG_RSP,
		narrow=x86_const.X86_REG_ESP,
		word=x86_const.X86_REG_SP,
		bytes=(x86_const.X86_REG_SPL,),
	),
	_X86Family(
		wide=x86_const.X86_REG_R8,
		narrow=x86_const.X86_REG_R8D,
		word=x86_const.X86_REG_R8W,
		bytes=(x86_const.X86_REG_R8B,),
	),
	_X86Family(
		wide=x86_const.X86_REG_R9,
		narrow=x86_const.X86_REG_R9D,
		word=x86_const.X86_REG_R9W,
		bytes=(x86_const.X86_REG_R9B,),
	),
	_X86Family(
		wide=x86_const.X86_REG_R10,
		narrow=x86_const.X86_REG_R10D,
		word=x86_const.X86_REG_R10W,
		bytes=(x86_const.X86_REG_R10B,),
	),
	_X86Family(
		wide=x86_const.X86_REG_R11,
		narrow=x86_const.X86_REG_R11D,
		word=x86_const.X86_REG_R11W,
		bytes=(x86_const.X86_REG_R11B,),
	),
	_X86Family(
		wide=x86_const.X86_REG_R12,
		narrow=x86_const.X86_REG_R12D,
		word=x86_const.X86_REG_R12W,
		bytes=(x86_const.X86_REG_R12B,),
	),
	_X86Family(
		wide=x86_const.X86_REG_R13,
		narrow=x86_const.X86_REG_R13D,
		word=x86_const.X86_REG_R13W,
		bytes=(x86_const.X86_REG_R13B,),
	),
	_X86Family(
		wide=x86_const.X86_REG_R14,
		narrow=x86_const.X86_REG_R14D,
		word=x86_const.X86_REG_R14W,
		bytes=(x86_const.X86_REG_R14B,),
	),
	_X86Family(
		wide=x86_const.X86_REG_R15,
		narrow=x86_const.X86_REG_R15D,
		word=x86_const.X86_REG_R15W,
		bytes=(x86_const.X86_REG_R15B,),
	),
)
X86_REGISTERS: Final[Mapping[int, X86Register]] = {
	register: X86Register(wide=family.wide, narrow=family.narrow, bits=bits)
	for family in _X86_FAMILIES
	for bits, registers in {
		64: (family.wide,),
		32: (family.narrow,),
		16: (family.word,),
		8: family.bytes,
	}.items()
	for register in registers
}
X86_CALLER_SAVED_64: Final = (
	x86_const.X86_REG_RAX,
	x86_const.X86_REG_RCX,
	x86_const.X86_REG_RDX,
	x86_const.X86_REG_RSI,
	x86_const.X86_REG_RDI,
	x86_const.X86_REG_R8,
	x86_const.X86_REG_R9,
	x86_const.X86_REG_R10,
	x86_const.X86_REG_R11,
)
X86_CALLER_SAVED_32: Final = (
	x86_const.X86_REG_EAX,
	x86_const.X86_REG_ECX,
	x86_const.X86_REG_EDX,
	x86_const.X86_REG_ESI,
	x86_const.X86_REG_EDI,
)
ARM_CALLER_SAVED: Final = (
	arm_const.ARM_REG_R0,
	arm_const.ARM_REG_R1,
	arm_const.ARM_REG_R2,
	arm_const.ARM_REG_R3,
	arm_const.ARM_REG_R12,
	arm_const.ARM_REG_LR,
)
X86_MOVES: Final = ("mov", "movabs")
X86_MEMORY_READERS: Final = frozenset(
	{
		"nop",
		"cmp",
		"test",
		"bt",
		"jmp",
		"ljmp",
		"call",
		"lcall",
		"div",
		"idiv",
		"mul",
		"imul",
		"cmpsb",
		"cmpsw",
		"cmpsd",
		"cmpsq",
		"prefetcht0",
		"prefetcht1",
		"prefetcht2",
		"prefetchnta",
		"prefetchw",
		"prefetchwt1",
		"clflush",
		"clflushopt",
		"clwb",
	}
)
X86_REPEATS: Final = frozenset({"rep", "repe", "repne", "repz", "repnz"})
X86_UNBOUNDED_STORES: Final = frozenset(
	{
		"fnsave",
		"fsave",
		"fnstenv",
		"fstenv",
		"fxsave",
		"fxsave64",
		"xsave",
		"xsave64",
		"xsavec",
		"xsavec64",
		"xsaveopt",
		"xsaveopt64",
		"xsaves",
		"xsaves64",
	}
)
ARM_WORD_MASK: Final = 0xFFFF_FFFF
ARM_MOVES: Final = frozenset(
	{arm_const.ARM_INS_MOV, arm_const.ARM_INS_MOVS, arm_const.ARM_INS_MOVW}
)
ARM_ADDITION_SIGNS: Final[Mapping[int, int]] = {
	arm_const.ARM_INS_ADD: 1,
	arm_const.ARM_INS_ADDW: 1,
	arm_const.ARM_INS_SUB: -1,
	arm_const.ARM_INS_SUBS: -1,
	arm_const.ARM_INS_SUBW: -1,
}
ARM_ZERO_EXTEND_MASKS: Final[Mapping[int, int]] = {
	arm_const.ARM_INS_UXTB: 0xFF,
	arm_const.ARM_INS_UXTH: 0xFFFF,
}
ARM_CONDITIONS: Final = frozenset(
	{
		arm_const.ARM_CC_EQ,
		arm_const.ARM_CC_NE,
		arm_const.ARM_CC_HS,
		arm_const.ARM_CC_LO,
		arm_const.ARM_CC_MI,
		arm_const.ARM_CC_PL,
		arm_const.ARM_CC_VS,
		arm_const.ARM_CC_VC,
		arm_const.ARM_CC_HI,
		arm_const.ARM_CC_LS,
		arm_const.ARM_CC_GE,
		arm_const.ARM_CC_LT,
		arm_const.ARM_CC_GT,
		arm_const.ARM_CC_LE,
	}
)
ARM_STORE_WIDTHS: Final[Mapping[int, int]] = {
	arm_const.ARM_INS_STR: 4,
	arm_const.ARM_INS_STRB: 1,
	arm_const.ARM_INS_STRH: 2,
	arm_const.ARM_INS_STRD: 8,
	arm_const.ARM_INS_STRT: 4,
	arm_const.ARM_INS_STRBT: 1,
	arm_const.ARM_INS_STRHT: 2,
	arm_const.ARM_INS_STL: 4,
	arm_const.ARM_INS_STLB: 1,
	arm_const.ARM_INS_STLH: 2,
	arm_const.ARM_INS_STREX: 4,
	arm_const.ARM_INS_STREXB: 1,
	arm_const.ARM_INS_STREXH: 2,
	arm_const.ARM_INS_STLEX: 4,
	arm_const.ARM_INS_STLEXB: 1,
	arm_const.ARM_INS_STLEXH: 2,
	arm_const.ARM_INS_VSTR: 8,
}
ARM_STORED_REGISTERS: Final[Mapping[int, int]] = {
	arm_const.ARM_INS_STR: 1,
	arm_const.ARM_INS_STRD: 2,
}
ARM_EXCLUSIVE_STORES: Final = frozenset(
	{
		arm_const.ARM_INS_STREX,
		arm_const.ARM_INS_STREXB,
		arm_const.ARM_INS_STREXH,
		arm_const.ARM_INS_STLEX,
		arm_const.ARM_INS_STLEXB,
		arm_const.ARM_INS_STLEXH,
	}
)
ARM_FLOATING_POINT_MULTIPLE_STORES: Final = frozenset(
	{arm_const.ARM_INS_VPUSH, arm_const.ARM_INS_VSTMIA, arm_const.ARM_INS_VSTMDB}
)
ARM_MULTIPLE_STORES: Final = frozenset(
	{
		arm_const.ARM_INS_PUSH,
		arm_const.ARM_INS_STM,
		arm_const.ARM_INS_STMDB,
		*ARM_FLOATING_POINT_MULTIPLE_STORES,
	}
)
ARM_DESCENDING_STORES: Final = frozenset(
	{
		arm_const.ARM_INS_PUSH,
		arm_const.ARM_INS_STMDB,
		arm_const.ARM_INS_VPUSH,
		arm_const.ARM_INS_VSTMDB,
	}
)
ARM_UNTRACKED_STORES: Final = frozenset(
	{
		arm_const.ARM_INS_STC,
		arm_const.ARM_INS_STC2,
		arm_const.ARM_INS_STC2L,
		arm_const.ARM_INS_STCL,
		arm_const.ARM_INS_STLEXD,
		arm_const.ARM_INS_STMDA,
		arm_const.ARM_INS_STMIB,
		arm_const.ARM_INS_STREXD,
		arm_const.ARM_INS_VST1,
		arm_const.ARM_INS_VST2,
		arm_const.ARM_INS_VST3,
		arm_const.ARM_INS_VST4,
	}
)
ARM_REGISTER_BYTES: Final[Mapping[str, int]] = {"s": 4, "d": 8, "q": 16}
ARM_LOAD_WIDTHS: Final[Mapping[int, int]] = {
	arm_const.ARM_INS_LDR: 4,
	arm_const.ARM_INS_LDRH: 2,
	arm_const.ARM_INS_LDRB: 1,
}
ARM_TRACKED_LOAD_WIDTHS: Final[Mapping[int, int]] = {
	arm_const.ARM_INS_LDR: 4,
	arm_const.ARM_INS_LDRH: 2,
}
X86_64_ARGUMENT_REGISTERS: Final = (
	x86_const.X86_REG_RDI,
	x86_const.X86_REG_RSI,
	x86_const.X86_REG_RDX,
	x86_const.X86_REG_RCX,
	x86_const.X86_REG_R8,
	x86_const.X86_REG_R9,
)
ARM_ARGUMENT_REGISTERS: Final = (
	arm_const.ARM_REG_R0,
	arm_const.ARM_REG_R1,
	arm_const.ARM_REG_R2,
	arm_const.ARM_REG_R3,
)
EM_386_STACK_ARGUMENTS: Final = 8
ARM_STACK_ARGUMENTS: Final = 8


def _register_arguments(machine: Machine) -> tuple[int, ...]:
	match machine:
		case Machine.EM_X86_64:
			return X86_64_ARGUMENT_REGISTERS
		case Machine.EM_386:
			return ()
		case Machine.EM_ARM:
			return ARM_ARGUMENT_REGISTERS
		case _ as unreachable:
			assert_never(unreachable)


REGISTER_ARGUMENTS: Final = {machine: _register_arguments(machine) for machine in Machine}
FLAG_REGISTER_NAMES: Final = frozenset(
	{
		"apsr",
		"apsr_nzcv",
		"apsr_nzcvq",
		"cpsr",
		"eflags",
		"flags",
		"fpscr",
		"fpscr_nzcv",
		"fpsw",
		"itstate",
		"rflags",
		"spsr",
	}
)


def is_data(instruction: CsInsn) -> bool:
	"""Whether the disassembler skipped these bytes as data rather than decoding an instruction."""
	return instruction.id == 0


def arm_predicated(instruction: CsInsn) -> bool:
	return (
		instruction.id not in (arm_const.ARM_INS_INVALID, arm_const.ARM_INS_IT)
		and instruction.cc in ARM_CONDITIONS
	)


def index_scale(machine: Machine, operand: CsOperand) -> int | None:
	"""What a memory operand's index register is multiplied by; None for a shift other than left."""
	match machine.family:
		case InstructionFamily.X86:
			return operand.mem.scale
		case InstructionFamily.ARM:
			return _arm_index_scale(cast("ArmCsOperand", operand))
		case _ as unreachable:
			assert_never(unreachable)


def _arm_index_scale(operand: ArmCsOperand) -> int | None:
	"""Capstone keeps an Arm index's shift and sign on the operand, not in ``mem.scale``."""
	match operand.shift.type:
		case arm_const.ARM_SFT_INVALID | arm_const.ARM_SFT_LSL:
			return (-1 if operand.subtracted else 1) << operand.shift.value
		case _:
			return None


def normalized(address: Address, machine: Machine) -> Address:
	"""Where the code at a function or call target address starts.

	An ARM address carries the Thumb bit, which is not part of where its code starts.

	>>> hex(normalized(Address(0x1001), Machine.EM_ARM))
	'0x1000'
	>>> hex(normalized(Address(0x1001), Machine.EM_X86_64))
	'0x1001'
	"""
	match machine.family:
		case InstructionFamily.ARM:
			return aligned(address)
		case InstructionFamily.X86:
			return address
		case _ as unreachable:
			assert_never(unreachable)


def disassemblers(core: ArmCore | None) -> Mapping[InstructionSet, Cs]:
	return {
		instruction_set: _disassembler(_architecture_and_mode(instruction_set, core))
		for instruction_set in InstructionSet
	}


def _architecture_and_mode(instruction_set: InstructionSet, core: ArmCore | None) -> CapstoneMode:
	"""An M-profile core's Thumb code has its own encodings, such as ``mrs r0, MSP``."""
	match instruction_set, core:
		case InstructionSet.T32, ArmCore(profile=ArmProfile.MICROCONTROLLER):
			return CapstoneMode(architecture=CS_ARCH_ARM, mode=CS_MODE_THUMB | CS_MODE_MCLASS)
		case _:
			return DISASSEMBLERS[instruction_set]


def _disassembler(mode: CapstoneMode) -> Cs:
	disassembler = Cs(mode.architecture, mode.mode)
	disassembler.detail = True
	disassembler.skipdata = True
	return disassembler


def program_counter(instruction: CsInsn, instruction_set: InstructionSet) -> int:
	"""The base a PC-relative operand of the instruction addresses from."""
	match instruction_set:
		case InstructionSet.A32:
			return instruction.address + 8
		case InstructionSet.T32:
			return (instruction.address + 4) & ~3
		case InstructionSet.X86_32 | InstructionSet.X86_64:
			return instruction.address + instruction.size
		case _ as unreachable:
			assert_never(unreachable)
