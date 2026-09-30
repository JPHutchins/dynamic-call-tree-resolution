# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""C compiler invocations for the fixture programs, and their runners."""

import subprocess
from itertools import islice
from pathlib import Path
from typing import IO, cast

FIXTURES = Path(__file__).parent / "fixtures"
ARM_HARNESS = FIXTURES / "arm" / "harness.c"
CORTEX_M3_HARNESS = FIXTURES / "cortex-m3"
CORTEX_A15_HARNESS = FIXTURES / "cortex-a15"
HOST_HARNESS = FIXTURES / "host"

_CC_FLAGS = ("-std=gnu2x", "-Wall", "-Wextra", "-Werror", "-Wdouble-promotion")


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
			"qemu-system-arm",
			"-machine",
			"lm3s6965evb",
			"-cpu",
			"cortex-m3",
			"-nographic",
			"-monitor",
			"none",
			"-serial",
			"stdio",
			"-icount",
			"shift=6,align=off,sleep=off",
			"-rtc",
			"clock=vm",
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
