# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Every README depth is tool output: each console block runs and matches, in order."""

import re
from pathlib import Path

import pytest

from tests.dctr import dctr
from tests.puncover_comparison import comparison

pytestmark = pytest.mark.image

REPOSITORY_ROOT = Path(__file__).parent.parent


def _console_blocks(readme: Path) -> list[list[str]]:
	return [
		str(block[1]).splitlines()
		for block in re.finditer(r"```console\n(.*?)```", readme.read_text(), re.DOTALL)
	]


def _transcripts(readme: Path) -> list[list[str]]:
	return [block for block in _console_blocks(readme) if block[0].startswith("$ dctr ")]


@pytest.mark.parametrize(
	"transcript",
	[
		pytest.param(transcript, id=transcript[0])
		for transcript in _transcripts(REPOSITORY_ROOT / "README.md")
	],
)
def test_readme_transcript_runs(zephyr_fixtures: Path, transcript: list[str]) -> None:
	command, *expected = transcript
	assert _matches(
		expected,
		dctr(
			*command.removeprefix("$ ").replace("$DCTR_FIXTURES", str(zephyr_fixtures)).split()[1:]
		).splitlines(),
	)


def _matches(expected: list[str], actual: list[str]) -> bool:
	if not expected:
		return True
	if expected[0] == "...":
		return any(_matches(expected[1:], actual[offset:]) for offset in range(len(actual) + 1))
	if not actual or expected[0] != actual[0]:
		return False
	return _matches(expected[1:], actual[1:])


def _prose(readme: Path) -> str:
	return re.sub(r"```console\n.*?```", "", readme.read_text(), flags=re.DOTALL)


def test_readme_prose_publishes_no_byte_count() -> None:
	assert [
		match[0] for match in re.finditer(r"\b\d+ bytes\b", _prose(REPOSITORY_ROOT / "README.md"))
	] == []


def test_readme_prose_quotes_only_sites_its_transcripts_print() -> None:
	printed = "\n".join(
		line for transcript in _transcripts(REPOSITORY_ROOT / "README.md") for line in transcript
	)
	assert [
		match[0]
		for match in re.finditer(r"\w+@0x[0-9a-f]+", _prose(REPOSITORY_ROOT / "README.md"))
		if match[0] not in printed
	] == []


def test_readme_puncover_comparison_is_the_generators_output(zephyr_fixtures: Path) -> None:
	(published,) = (
		block[1:]
		for block in _console_blocks(REPOSITORY_ROOT / "README.md")
		if block[0] == "$ uv run python -m tests.puncover_comparison"
	)
	assert published == comparison(zephyr_fixtures).splitlines()


def test_the_puncover_comparison_needs_the_arm_toolchain_on_path(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setenv("PATH", str(tmp_path))
	with pytest.raises(RuntimeError, match="nix develop"):
		comparison(tmp_path)
