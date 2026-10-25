# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""C compiler invocations for the fixture programs, and their runners."""

from __future__ import annotations

import shlex
import subprocess
from enum import StrEnum
from itertools import islice
from pathlib import Path
from typing import IO, TYPE_CHECKING, assert_never, cast

from salix import Struct

from dynamic_call_tree_resolution import Address

if TYPE_CHECKING:
	from collections.abc import Mapping

FIXTURES = Path(__file__).parent / "fixtures"
ARM_HARNESS = FIXTURES / "arm" / "harness.c"
CORTEX_M3_HARNESS = FIXTURES / "cortex-m3"
CORTEX_A15_HARNESS = FIXTURES / "cortex-a15"
HOST_HARNESS = FIXTURES / "host"

_CC_FLAGS = ("-std=gnu2x", "-Wall", "-Wextra", "-Werror", "-Wdouble-promotion")


class ZephyrMachine(StrEnum):
	"""A QEMU machine a Zephyr testbed runs on."""

	LM3S6965EVB = "lm3s6965evb"
	MPS2_AN385 = "mps2-an385"
	MPS3_AN536 = "mps3-an536"

	@property
	def cpu(self) -> str:
		match self:
			case ZephyrMachine.LM3S6965EVB | ZephyrMachine.MPS2_AN385:
				return "cortex-m3"
			case ZephyrMachine.MPS3_AN536:
				return "cortex-r52"
			case _ as unreachable:
				assert_never(unreachable)

	@property
	def clock(self) -> tuple[str, ...]:
		"""How QEMU keeps its time: by counting instructions, except where gdb cannot resume from
		a breakpoint inside an interrupt handler under that clock."""
		match self:
			case ZephyrMachine.LM3S6965EVB | ZephyrMachine.MPS2_AN385:
				return ("-icount", "shift=6,align=off,sleep=off", "-rtc", "clock=vm")
			case ZephyrMachine.MPS3_AN536:
				return ()
			case _ as unreachable:
				assert_never(unreachable)


def _zephyr_qemu(machine: ZephyrMachine) -> tuple[str, ...]:
	return (
		"qemu-system-arm",
		"-machine",
		machine,
		"-cpu",
		machine.cpu,
		"-monitor",
		"none",
		*machine.clock,
	)


def host_cc(*flags: str) -> tuple[str, ...]:
	return ("cc", *_CC_FLAGS, *flags)


def cortex_m3_cc(*flags: str) -> tuple[str, ...]:
	return (
		"arm-none-eabi-gcc",
		"-mcpu=cortex-m3",
		"-mthumb",
		*_CC_FLAGS,
		"-ffreestanding",
		"-nostdlib",
		f"-I{FIXTURES}",
		f"-T{CORTEX_M3_HARNESS / 'cortex_m3.ld'}",
		str(ARM_HARNESS),
		*flags,
	)


def cortex_a15_cc(*flags: str) -> tuple[str, ...]:
	return (
		"arm-none-eabi-gcc",
		"-mcpu=cortex-a15",
		"-marm",
		"-mno-unaligned-access",
		*_CC_FLAGS,
		"-ffreestanding",
		"-nostdlib",
		f"-I{FIXTURES}",
		f"-T{CORTEX_A15_HARNESS / 'cortex_a15.ld'}",
		str(ARM_HARNESS),
		*flags,
	)


def build_cortex_a15(sources: tuple[Path, ...], output: Path, *flags: str) -> Path:
	subprocess.run(
		[*cortex_a15_cc(*flags), *map(str, sources), "-o", str(output)],
		check=True,
		capture_output=True,
	)
	return output


def build_cortex_m3(sources: tuple[Path, ...], output: Path, *flags: str) -> Path:
	subprocess.run(
		[*cortex_m3_cc(*flags), *map(str, sources), "-o", str(output)],
		check=True,
		capture_output=True,
	)
	return output


def build_host(sources: tuple[Path, ...], output: Path, *flags: str) -> Path:
	subprocess.run(
		[
			*host_cc(f"-I{FIXTURES}", str(HOST_HARNESS / "harness.c"), *flags),
			*map(str, sources),
			"-o",
			str(output),
		],
		check=True,
		capture_output=True,
	)
	return output


def run_host(executable: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
	return subprocess.run(
		[str(executable), *arguments],
		check=False,
		capture_output=True,
		text=True,
		timeout=30,
	)


def run_cortex_m3(elf: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
	return _run_qemu("lm3s6965evb", "cortex-m3", elf, arguments)


def run_cortex_a15(elf: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
	return _run_qemu("virt", "cortex-a15", elf, arguments)


def zephyr_console_cortex_m3(elf: Path, line_count: int) -> tuple[str, ...]:
	with subprocess.Popen(
		[
			"timeout",
			"30",
			*_zephyr_qemu(ZephyrMachine.LM3S6965EVB),
			"-nographic",
			"-serial",
			"stdio",
			"-kernel",
			str(elf),
		],
		stdin=subprocess.DEVNULL,
		stdout=subprocess.PIPE,
		encoding="utf-8",
	) as qemu:
		lines = tuple(
			line.rstrip("\n") for line in islice(cast("IO[str]", qemu.stdout), line_count)
		)
		qemu.terminate()
		return lines


def zephyr_stack_pointers(
	machine: ZephyrMachine, elf: Path, breakpoint: str, hits: int
) -> tuple[int, ...]:
	"""The stack pointer at each of a breakpoint's first hits, under gdb on QEMU."""
	return tuple(
		int(stack_pointer, 16)
		for line in subprocess.run(
			[
				"timeout",
				"-k",
				"5",
				"60",
				"arm-none-eabi-gdb",
				"-batch",
				"-nx",
				"-ex",
				"target remote | "
				+ shlex.join(
					(
						*_zephyr_qemu(machine),
						"-display",
						"none",
						"-serial",
						"null",
						"-kernel",
						str(elf),
						"-S",
						"-gdb",
						"stdio",
					)
				),
				"-ex",
				f"break {breakpoint}",
				*(
					argument
					for _ in range(hits)
					for argument in ("-ex", "continue", "-ex", r'printf "hit %#x\n", $sp')
				),
				"-ex",
				"kill",
				str(elf),
			],
			check=False,
			capture_output=True,
			text=True,
		).stdout.splitlines()
		if line.startswith("hit ")
		for _, stack_pointer in (line.split(),)
	)


def zephyr_values(
	machine: ZephyrMachine, elf: Path, breakpoints: tuple[str, ...], expressions: tuple[str, ...]
) -> tuple[int, ...]:
	"""Each expression's value at each breakpoint's first hit, in order, under gdb on QEMU."""
	return tuple(
		int(line.split()[1], 16)
		for line in subprocess.run(
			[
				"timeout",
				"-k",
				"5",
				"60",
				"arm-none-eabi-gdb",
				"-batch",
				"-nx",
				"-ex",
				"target remote | "
				+ shlex.join(
					(
						*_zephyr_qemu(machine),
						"-display",
						"none",
						"-serial",
						"null",
						"-kernel",
						str(elf),
						"-S",
						"-gdb",
						"stdio",
					)
				),
				*(
					argument
					for breakpoint in breakpoints
					for argument in (
						"-ex",
						"delete",
						"-ex",
						f"break {breakpoint}",
						"-ex",
						"continue",
						*(
							argument
							for expression in expressions
							for argument in ("-ex", rf'printf "value %#x\n", {expression}')
						),
					)
				),
				"-ex",
				"kill",
				str(elf),
			],
			check=False,
			capture_output=True,
			text=True,
		).stdout.splitlines()
		if line.startswith("value ")
	)


class IndirectCall(Struct):
	"""One indirect call QEMU ran."""

	site: Address
	target: Address
	thread: Address
	"""The address of the ``struct k_thread`` that was current."""


def zephyr_indirect_calls_cortex_m3(
	elf: Path, registers: Mapping[Address, str], script: Path, seconds: int, hits: int
) -> frozenset[IndirectCall]:
	"""Each indirect call QEMU runs at a site, read from the register the site branches through,
	under gdb until the seconds or the hits run out."""
	script.write_text(
		"set pagination off\nset confirm off\nset $hits = 0\n"
		"target remote | "
		+ shlex.join(
			(
				*_zephyr_qemu(ZephyrMachine.LM3S6965EVB),
				"-display",
				"none",
				"-serial",
				"null",
				"-kernel",
				str(elf),
				"-S",
				"-gdb",
				"stdio",
			)
		)
		+ "\n"
		+ "".join(
			f"break *{site:#x}\ncommands\nsilent\n"
			f'printf "call %#x %#x %#x\\n", $pc, ${register}, '
			"(unsigned)_kernel.cpus[0].current\n"
			f"set $hits = $hits + 1\nif $hits >= {hits}\nkill\nquit\nend\ncontinue\nend\n"
			for site, register in registers.items()
		)
		+ "continue\nkill\nquit\n"
	)
	return frozenset(
		IndirectCall(
			site=Address(int(site, 16)),
			target=Address(int(target, 16)),
			thread=Address(int(thread, 16)),
		)
		for line in subprocess.run(
			[
				"timeout",
				str(seconds),
				"arm-none-eabi-gdb",
				"-batch",
				"-nx",
				"-x",
				str(script),
				str(elf),
			],
			check=False,
			capture_output=True,
			text=True,
		).stdout.splitlines()
		if line.startswith("call ")
		for _, site, target, thread in (line.split(),)
	)


def _run_qemu(
	machine: str, cpu: str, elf: Path, arguments: tuple[str, ...]
) -> subprocess.CompletedProcess[str]:
	return subprocess.run(
		[
			"qemu-system-arm",
			"-machine",
			machine,
			"-cpu",
			cpu,
			"-nographic",
			"-monitor",
			"none",
			"-serial",
			"none",
			"-chardev",
			"stdio,id=semihosting",
			"-semihosting-config",
			",".join(
				(
					"enable=on",
					"target=native",
					"chardev=semihosting",
					*(f"arg={argument}" for argument in (elf.name, *arguments)),
				)
			),
			"-kernel",
			str(elf),
		],
		check=False,
		capture_output=True,
		text=True,
		timeout=30,
	)
