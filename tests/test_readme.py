# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The README usage block is a transcript: run it and match, in order."""

import re
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parent.parent


def test_readme_usage_runs() -> None:
	usage = _usage_block(REPOSITORY_ROOT / "README.md")
	command = next(line for line in usage if line.startswith("$ "))
	executable = Path(sys.executable).with_name("dctr")
	result = subprocess.run(
		[str(executable), *command.removeprefix("$ ").split()[1:]],
		check=True,
		capture_output=True,
		text=True,
		cwd=REPOSITORY_ROOT,
	)
	assert _matches(usage[1:], result.stdout.splitlines())


def _usage_block(readme: Path) -> list[str]:
	block = re.search(r"```console\n(.*?)```", readme.read_text(), re.DOTALL)
	assert block is not None
	return block.group(1).splitlines()


def _matches(expected: list[str], actual: list[str]) -> bool:
	if not expected:
		return True
	if expected[0] == "...":
		return any(_matches(expected[1:], actual[offset:]) for offset in range(len(actual) + 1))
	if not actual or expected[0] != actual[0]:
		return False
	return _matches(expected[1:], actual[1:])
