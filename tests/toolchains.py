# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""C compiler invocations for the fixture programs, and the Cortex-M3 QEMU runner."""

import subprocess
from pathlib import Path

CORTEX_M3_HARNESS = Path(__file__).parent / "fixtures" / "cortex-m3"

_CC_FLAGS = ("-std=gnu2x", "-Wall", "-Wextra", "-Werror", "-Wdouble-promotion")


def host_cc(*flags: str) -> tuple[str, ...]:
	"""One host cc invocation with the fixture warning regime, plus extra flags."""
	return ("cc", *_CC_FLAGS, *flags)


def cortex_m3_cc(*flags: str) -> tuple[str, ...]:
	"""One bare-metal Cortex-M3 link of the harness with the fixture warning regime."""
	return (
		"arm-none-eabi-gcc",
		"-mcpu=cortex-m3",
		"-mthumb",
		*_CC_FLAGS,
		"-ffreestanding",
		"-nostdlib",
		f"-I{CORTEX_M3_HARNESS}",
		f"-T{CORTEX_M3_HARNESS / 'cortex_m3.ld'}",
		str(CORTEX_M3_HARNESS / "harness.c"),
		*flags,
	)


def build_cortex_m3(sources: tuple[Path, ...], output: Path, *flags: str) -> Path:
	"""Link ``sources`` with the harness into a bare-metal Cortex-M3 image."""
	subprocess.run(
		[*cortex_m3_cc(*flags), *map(str, sources), "-o", str(output)],
		check=True,
		capture_output=True,
	)
	return output


def run_cortex_m3(elf: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
	"""Run a harness image on QEMU.

	The semihosting command line becomes ``argv``, with the image's file name
	as ``argv[0]``; ``main``'s return value becomes the exit status.
	"""
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
