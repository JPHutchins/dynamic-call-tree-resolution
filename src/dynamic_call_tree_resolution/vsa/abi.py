# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Per-machine registers and instruction classes the analysis models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from capstone import (
	CS_ARCH_ARM,
	CS_ARCH_X86,
	CS_MODE_32,
	CS_MODE_64,
	CS_MODE_THUMB,
	arm_const,
	x86_const,
)

from dynamic_call_tree_resolution.model import Address, Machine, aligned

if TYPE_CHECKING:
	from collections.abc import Mapping


DISASSEMBLERS: Final[Mapping[Machine, tuple[int, int]]] = {
	Machine.EM_X86_64: (CS_ARCH_X86, CS_MODE_64),
	Machine.EM_386: (CS_ARCH_X86, CS_MODE_32),
	Machine.EM_ARM: (CS_ARCH_ARM, CS_MODE_THUMB),
}

SP_REGISTERS: Final[Mapping[Machine, tuple[int, ...]]] = {
	Machine.EM_X86_64: (x86_const.X86_REG_RSP, x86_const.X86_REG_RBP),
	Machine.EM_386: (x86_const.X86_REG_ESP, x86_const.X86_REG_EBP),
	Machine.EM_ARM: (arm_const.ARM_REG_SP,),
}

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
ARM_LOADS: Final = ("ldr", "ldr.w", "ldr.n")
ARM_MOVES: Final = ("mov", "movs", "mov.w")
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


def normalized(address: Address, machine: Machine) -> Address:
	return aligned(address) if machine is Machine.EM_ARM else address
