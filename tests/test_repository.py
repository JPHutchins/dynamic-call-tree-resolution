# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Firmware builds come from `nix build .#fixtures`, never from files committed here."""

import subprocess
from pathlib import Path

import pytest

from tests.nix_fixtures import zephyr_fixtures_root

REPOSITORY_ROOT = Path(__file__).parent.parent


def _tracked(directory: str) -> list[Path]:
	return [
		REPOSITORY_ROOT / path
		for path in subprocess.run(
			["git", "ls-files", "-z", directory],
			check=True,
			capture_output=True,
			text=True,
			cwd=REPOSITORY_ROOT,
		).stdout.split("\0")
		if path
	]


def _is_build_output(path: Path) -> bool:
	if path.suffix in {".su", ".ci"}:
		return True
	with path.open("rb") as stream:
		return stream.read(4) == b"\x7fELF"


def test_no_build_output_is_committed_under_tests_fixtures() -> None:
	assert [
		path.relative_to(REPOSITORY_ROOT).as_posix()
		for path in _tracked("tests/fixtures")
		if _is_build_output(path)
	] == []


@pytest.mark.parametrize(
	("name", "content", "build_output"),
	[
		("main.c.su", b"main.c:49:5:main\t0\tstatic\n", True),
		("main.c.ci", b'graph: { title: "main.c" }\n', True),
		("zephyr", b"\x7fELF\x01\x01\x01", True),
		("main.c", b"int main(void) {}\n", False),
	],
)
def test_build_output_is_recognized_by_suffix_or_elf_magic(
	tmp_path: Path, name: str, content: bytes, build_output: bool
) -> None:
	(tmp_path / name).write_bytes(content)
	assert _is_build_output(tmp_path / name) is build_output


@pytest.mark.parametrize("environment", [{}, {"DCTR_FIXTURES": ""}], ids=["unset", "empty"])
def test_image_tests_fail_naming_dctr_fixtures_when_it_is_unset(
	environment: dict[str, str],
) -> None:
	with pytest.raises(pytest.fail.Exception, match="DCTR_FIXTURES is unset"):
		zephyr_fixtures_root(environment)
