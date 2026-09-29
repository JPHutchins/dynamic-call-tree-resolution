# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""C compiler invocations for the fixture programs, and their runners."""

import subprocess
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
CORTEX_M3_HARNESS = FIXTURES / "cortex-m3"
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
		str(CORTEX_M3_HARNESS / "harness.c"),
		*flags,
	)


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
	return subprocess.run(
		[
			"qemu-system-arm",
			"-machine",
			"lm3s6965evb",
			"-cpu",
			"cortex-m3",
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
