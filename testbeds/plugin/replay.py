# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Replay a Zephyr build's C compiles through the descriptors plugin.

Each translation unit is compiled twice by Arm GNU, without and with the plugin; the replay fails
unless both disassemble the same.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final, cast

COMPILER: Final = "arm-none-eabi-gcc"
OBJDUMP: Final = "arm-none-eabi-objdump"


def main(build: Path, plugin: Path, root: Path) -> None:
	compiler_includes = tuple(
		_run((COMPILER, f"-print-file-name={name}"), root).strip()
		for name in ("include", "include-fixed")
	)
	with tempfile.TemporaryDirectory() as scratch, ThreadPoolExecutor(os.cpu_count()) as pool:
		(build / "zephyr" / "descriptors.txt").write_text(
			"".join(
				f"{record}\n"
				for record in sorted(
					record
					for replay in tuple(
						pool.submit(
							_descriptors,
							entry,
							Path(scratch) / f"{index}.o",
							build=build,
							plugin=plugin,
							root=root,
							compiler_includes=compiler_includes,
						)
						for index, entry in enumerate(
							entry
							for entry in cast(
								"list[dict[str, str]]",
								json.loads((build / "compile_commands.json").read_text()),
							)
							if entry["file"].endswith(".c")
						)
					)
					for record in replay.result()
				)
			)
		)


def _descriptors(
	entry: dict[str, str],
	output: Path,
	*,
	build: Path,
	plugin: Path,
	root: Path,
	compiler_includes: tuple[str, ...],
) -> tuple[str, ...]:
	compile_command = _replayed(tuple(shlex.split(entry["command"])), output, compiler_includes)
	_run(compile_command, entry["directory"])
	plain = _run((OBJDUMP, "-d", output.name), output.parent)
	records = _run((*compile_command, f"-fplugin={plugin}"), entry["directory"])
	if _run((OBJDUMP, "-d", output.name), output.parent) != plain:
		raise RuntimeError(f"the plugin changed the code of {entry['file']}")
	return tuple(_record(line, _unit(entry, build), root) for line in records.splitlines())


def _record(line: str, unit: str, root: Path) -> str:
	kind, function, file, *rest = line.split("\t")
	return "\t".join((kind, unit, function, _relative(file, root), *rest))


def _unit(entry: dict[str, str], build: Path) -> str:
	return (
		(Path(entry["directory"]) / entry["output"]).relative_to(build).with_suffix("").as_posix()
	)


def _relative(path: str, root: Path) -> str:
	return (
		os.path.relpath(os.path.normpath(path), root)
		if Path(os.path.normpath(path)).is_relative_to(root)
		else path
	)


def _replayed(
	arguments: tuple[str, ...], output: Path, compiler_includes: tuple[str, ...]
) -> tuple[str, ...]:
	return (
		COMPILER,
		"-nostdinc",
		*(
			argument
			for index, argument in enumerate(arguments)
			if index > 0
			and argument != "-o"
			and arguments[index - 1] != "-o"
			and argument != "-fstack-usage"
			and not argument.startswith("-fcallgraph-info")
		),
		*(
			include
			for directory in (
				*compiler_includes,
				*(
					f"{argument.removeprefix('--sysroot=')}/{name}"
					for argument in arguments
					if argument.startswith("--sysroot=")
					for name in ("sys-include", "include")
				),
			)
			for include in ("-isystem", directory)
		),
		"-o",
		str(output),
	)


def _run(command: tuple[str, ...], directory: str | Path) -> str:
	completed = subprocess.run(command, cwd=directory, capture_output=True, text=True, check=False)
	if completed.returncode != 0:
		raise RuntimeError(f"{shlex.join(command)}\n{completed.stderr}")
	return completed.stdout


if __name__ == "__main__":
	main(*(Path(argument).resolve() for argument in sys.argv[1:4]))
