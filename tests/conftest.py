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


def _compile_multi(source_directory: Path, build_directory: Path) -> Path:
	definition_object = build_directory / "multi_def.o"
	use_object = build_directory / "multi_use.o"
	output = build_directory / "multi.elf"
	subprocess.run(
		["cc", "-c", "-O0", str(source_directory / "multi_def.c"), "-o", str(definition_object)],
		check=True,
		capture_output=True,
	)
	subprocess.run(
		["cc", "-c", "-g", "-O0", str(source_directory / "multi_use.c"), "-o", str(use_object)],
		check=True,
		capture_output=True,
	)
	subprocess.run(
		["cc", "-no-pie", str(definition_object), str(use_object), "-o", str(output)],
		check=True,
		capture_output=True,
	)
	return output


@pytest.fixture(scope="session")
def fixture_elfs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
	"""Compile every fixture variant once per session."""
	build_directory = tmp_path_factory.mktemp("elfs")
	variants = {
		variant: _compile(
			FIXTURE_DIRECTORY / source,
			build_directory / f"{source.removesuffix('.c')}.{variant}.elf",
			*flags,
		)
		for variant, (source, flags) in FIXTURE_VARIANTS.items()
	}
	variants["multi"] = _compile_multi(FIXTURE_DIRECTORY, build_directory)
	return variants
