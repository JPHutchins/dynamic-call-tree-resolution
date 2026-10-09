# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Per-machine registers and instruction classes the analysis models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, assert_never

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

	from capstone import CsInsn


DISASSEMBLERS: Final[Mapping[InstructionSet, tuple[int, int]]] = {
	InstructionSet.X86_64: (CS_ARCH_X86, CS_MODE_64),
	InstructionSet.X86_32: (CS_ARCH_X86, CS_MODE_32),
	InstructionSet.A32: (CS_ARCH_ARM, CS_MODE_ARM),
	InstructionSet.T32: (CS_ARCH_ARM, CS_MODE_THUMB),
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
ARM_TRANSFERS: Final = ("bl", "blx", "bx", "b", "pop", "svc", "bkpt", "udf", "tbb", "tbh")
X86_CALLS: Final = ("call",)
X86_RETURNING_TRAPS: Final = ("syscall", "int", "int3", "hlt")
ARM_RETURNING_TRAPS: Final = ("svc", "bkpt")
ARM_CALLS: Final = ("bl", "blx")
ARM_CONDITIONAL: Final = frozenset(
	{
		"beq",
		"bne",
		"bcs",
		"bcc",
		"bmi",
		"bpl",
		"bvs",
		"bvc",
		"bhi",
		"bls",
		"bge",
		"blt",
		"bgt",
		"ble",
		"bhs",
		"blo",
		"cbz",
		"cbnz",
	}
)


class X86Register(Struct):
	"""Where an x86 general-purpose register sits in its family."""

	wide: int
	"""The family's 64-bit register."""
	narrow: int
	"""The family's 32-bit register."""
	bits: int


_X86_FAMILIES: Final = (
	(
		x86_const.X86_REG_RAX,
		x86_const.X86_REG_EAX,
		x86_const.X86_REG_AX,
		x86_const.X86_REG_AL,
		x86_const.X86_REG_AH,
	),
	(
		x86_const.X86_REG_RBX,
		x86_const.X86_REG_EBX,
		x86_const.X86_REG_BX,
		x86_const.X86_REG_BL,
		x86_const.X86_REG_BH,
	),
	(
		x86_const.X86_REG_RCX,
		x86_const.X86_REG_ECX,
		x86_const.X86_REG_CX,
		x86_const.X86_REG_CL,
		x86_const.X86_REG_CH,
	),
	(
		x86_const.X86_REG_RDX,
		x86_const.X86_REG_EDX,
		x86_const.X86_REG_DX,
		x86_const.X86_REG_DL,
		x86_const.X86_REG_DH,
	),
	(x86_const.X86_REG_RSI, x86_const.X86_REG_ESI, x86_const.X86_REG_SI, x86_const.X86_REG_SIL),
	(x86_const.X86_REG_RDI, x86_const.X86_REG_EDI, x86_const.X86_REG_DI, x86_const.X86_REG_DIL),
	(x86_const.X86_REG_RBP, x86_const.X86_REG_EBP, x86_const.X86_REG_BP, x86_const.X86_REG_BPL),
	(x86_const.X86_REG_RSP, x86_const.X86_REG_ESP, x86_const.X86_REG_SP, x86_const.X86_REG_SPL),
	(x86_const.X86_REG_R8, x86_const.X86_REG_R8D, x86_const.X86_REG_R8W, x86_const.X86_REG_R8B),
	(x86_const.X86_REG_R9, x86_const.X86_REG_R9D, x86_const.X86_REG_R9W, x86_const.X86_REG_R9B),
	(x86_const.X86_REG_R10, x86_const.X86_REG_R10D, x86_const.X86_REG_R10W, x86_const.X86_REG_R10B),
	(x86_const.X86_REG_R11, x86_const.X86_REG_R11D, x86_const.X86_REG_R11W, x86_const.X86_REG_R11B),
	(x86_const.X86_REG_R12, x86_const.X86_REG_R12D, x86_const.X86_REG_R12W, x86_const.X86_REG_R12B),
	(x86_const.X86_REG_R13, x86_const.X86_REG_R13D, x86_const.X86_REG_R13W, x86_const.X86_REG_R13B),
	(x86_const.X86_REG_R14, x86_const.X86_REG_R14D, x86_const.X86_REG_R14W, x86_const.X86_REG_R14B),
	(x86_const.X86_REG_R15, x86_const.X86_REG_R15D, x86_const.X86_REG_R15W, x86_const.X86_REG_R15B),
)
X86_REGISTERS: Final[Mapping[int, X86Register]] = {
	register: X86Register(wide=wide, narrow=narrow, bits=bits)
	for wide, narrow, word, *bytes_ in _X86_FAMILIES
	for register, bits in ((wide, 64), (narrow, 32), (word, 16), *((byte, 8) for byte in bytes_))
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
ARM_MOVES: Final = ("mov", "movs")
ARM_ZERO_EXTEND_MASKS: Final[Mapping[str, int]] = {"uxtb": 0xFF, "uxth": 0xFFFF}
ARM_CONDITION_SUFFIXES: Final[Mapping[int, str]] = {
	arm_const.ARM_CC_EQ: "eq",
	arm_const.ARM_CC_NE: "ne",
	arm_const.ARM_CC_HS: "hs",
	arm_const.ARM_CC_LO: "lo",
	arm_const.ARM_CC_MI: "mi",
	arm_const.ARM_CC_PL: "pl",
	arm_const.ARM_CC_VS: "vs",
	arm_const.ARM_CC_VC: "vc",
	arm_const.ARM_CC_HI: "hi",
	arm_const.ARM_CC_LS: "ls",
	arm_const.ARM_CC_GE: "ge",
	arm_const.ARM_CC_LT: "lt",
	arm_const.ARM_CC_GT: "gt",
	arm_const.ARM_CC_LE: "le",
}
ARM_STORE_WIDTHS: Final[Mapping[str, int]] = {
	"str": 4,
	"strb": 1,
	"strh": 2,
	"strd": 8,
	"strt": 4,
	"strbt": 1,
	"strht": 2,
	"stl": 4,
	"stlb": 1,
	"stlh": 2,
	"strex": 4,
	"strexb": 1,
	"strexh": 2,
	"stlex": 4,
	"stlexb": 1,
	"stlexh": 2,
	"vstr": 8,
}
ARM_STORED_REGISTERS: Final[Mapping[str, int]] = {"str": 1, "strd": 2}
ARM_EXCLUSIVE_STORES: Final = frozenset({"strex", "strexb", "strexh", "stlex", "stlexb", "stlexh"})
ARM_MULTIPLE_STORES: Final = frozenset(
	{"push", "stm", "stmia", "stmea", "stmdb", "stmfd", "vpush", "vstm", "vstmia", "vstmdb"}
)
ARM_DESCENDING_STORES: Final = frozenset({"push", "stmdb", "stmfd", "vpush", "vstmdb"})
ARM_REGISTER_BYTES: Final[Mapping[str, int]] = {"s": 4, "d": 8, "q": 16}
ARM_LOAD_WIDTHS: Final[Mapping[str, int]] = {"ldr": 4, "ldrh": 2, "ldrb": 1}
ARM_TRACKED_LOAD_WIDTHS: Final[Mapping[str, int]] = {"ldr": 4, "ldrh": 2}
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


def arm_mnemonic(instruction: CsInsn) -> str:
	unqualified = instruction.mnemonic.split(".")[0]
	if instruction.mnemonic == ".byte":
		return unqualified
	return unqualified.removesuffix(ARM_CONDITION_SUFFIXES.get(instruction.cc, ""))


def arm_predicated(instruction: CsInsn) -> bool:
	return arm_mnemonic(instruction) != instruction.mnemonic.split(".")[0]


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
		instruction_set: _disassembler(*_architecture_and_mode(instruction_set, core))
		for instruction_set in InstructionSet
	}


def _architecture_and_mode(
	instruction_set: InstructionSet, core: ArmCore | None
) -> tuple[int, int]:
	"""An M-profile core's Thumb code has its own encodings, such as ``mrs r0, MSP``."""
	match instruction_set, core:
		case InstructionSet.T32, ArmCore(profile=ArmProfile.MICROCONTROLLER):
			return CS_ARCH_ARM, CS_MODE_THUMB | CS_MODE_MCLASS
		case _:
			return DISASSEMBLERS[instruction_set]


def _disassembler(architecture: int, mode: int) -> Cs:
	disassembler = Cs(architecture, mode)
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
