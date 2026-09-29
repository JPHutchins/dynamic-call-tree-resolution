# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Every README console block is a transcript: run it and match, in order."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.image

REPOSITORY_ROOT = Path(__file__).parent.parent


def _transcripts(readme: Path) -> list[list[str]]:
	return [
		str(block[1]).splitlines()
		for block in re.finditer(r"```console\n(.*?)```", readme.read_text(), re.DOTALL)
	]


@pytest.mark.parametrize(
	"transcript",
	[
		pytest.param(transcript, id=transcript[0])
		for transcript in _transcripts(REPOSITORY_ROOT / "README.md")
	],
)
def test_readme_transcript_runs(transcript: list[str]) -> None:
	command, *expected = transcript
	executable = Path(sys.executable).with_name("dctr")
	result = subprocess.run(
		[str(executable), *command.removeprefix("$ ").split()[1:]],
		check=True,
		capture_output=True,
		text=True,
		cwd=REPOSITORY_ROOT,
	)
	assert _matches(expected, result.stdout.splitlines())


def _matches(expected: list[str], actual: list[str]) -> bool:
	if not expected:
		return True
	if expected[0] == "...":
		return any(_matches(expected[1:], actual[offset:]) for offset in range(len(actual) + 1))
	if not actual or expected[0] != actual[0]:
		return False
	return _matches(expected[1:], actual[1:])
