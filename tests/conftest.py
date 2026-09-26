# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Pytest fixtures compiling the C fixture programs."""

import subprocess
from pathlib import Path

import pytest

FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures"

FIXTURE_VARIANTS: dict[str, tuple[str, tuple[str, ...]]] = {
	"pie": ("device_model.c", ("-g", "-O0")),
	"nopie": ("device_model.c", ("-g", "-O0", "-no-pie")),
	"nodebug": ("device_model.c", ("-O0", "-no-pie")),
	"o2": ("device_model.c", ("-g", "-O2", "-no-pie")),
	"dwarf4": ("device_model.c", ("-g", "-gdwarf-4", "-O0", "-no-pie")),
	"minimal.pie": ("minimal.c", ("-g", "-O0")),
}


def _compile(source: Path, output: Path, *flags: str) -> Path:
	subprocess.run(
		["cc", *flags, str(source), "-o", str(output)],
		check=True,
		capture_output=True,
	)
	return output


@pytest.fixture(scope="session")
def fixture_elfs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
	"""Compile every fixture variant once per session."""
	build_directory = tmp_path_factory.mktemp("elfs")
	return {
		variant: _compile(
			FIXTURE_DIRECTORY / source,
			build_directory / f"{source.removesuffix('.c')}.{variant}.elf",
			*flags,
		)
		for variant, (source, flags) in FIXTURE_VARIANTS.items()
	}
