# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The ``descriptors`` command."""

import sys
from pathlib import Path  # cyclopts evaluates Annotated[Path, ...] at runtime
from typing import Annotated

from cyclopts import Parameter

from dynamic_call_tree_resolution.cli.app import app
from dynamic_call_tree_resolution.gcc_dumps import dump_descriptors


@app.command  # type: ignore[misc]
def descriptors(
	build_directory: Annotated[
		Path,
		Parameter(help="build directory with compile_commands.json, its objects and their dumps"),
	],
	*,
	root: Annotated[Path, Parameter(help="write source paths relative to this directory")] = Path(),
	output: Annotated[
		Path | None,
		Parameter(
			help="file to write; BUILD_DIRECTORY/zephyr/descriptors.txt, where --narrow-by-field "
			"reads it, by default"
		),
	] = None,
) -> None:
	"""Write a build's descriptors.txt from the alias dumps its own GCC wrote, without a plugin.

	Build every C unit with -fdump-tree-alias-lineno, -g and optimization: GCC runs the alias
	pass, and so dumps a function, only when optimizing it, and writes no dump for a unit without
	a function. Each unit's object supplies the types its dump names. A function without a dump,
	or a store of a function this cannot place, fails rather than leave a target out. Checked
	against GCC 14.3: the Zephyr SDK 1.0.1's 14.3.0 and Arm GNU 14.3.1, whose descriptors plugin
	writes the same records.

	Raises:
		ValueError: for a store of a function into a record the unit's types do not describe.
	"""
	described = dump_descriptors(build_directory, root.resolve())
	for gap in described.gaps:
		print(
			f"no record for {gap.at.file}:{gap.at.line}:{gap.at.column} in {gap.at.function}: "
			f"{gap.statement}",
			file=sys.stderr,
		)
	if described.gaps:
		raise ValueError(
			f"{len(described.gaps)} stores of a function have no record, so descriptors.txt "
			"would leave their targets out: none was written"
		)
	(output or build_directory / "zephyr" / "descriptors.txt").write_text(
		"".join(f"{line}\n" for line in described.lines)
	)
	print(
		f"{len(described.lines)} records from the alias dumps of {described.dumps} of "
		f"{described.units} C units",
		file=sys.stderr,
	)
