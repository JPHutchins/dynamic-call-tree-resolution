# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Pytest fixtures compiling the C fixture programs, and locating the Zephyr ones."""

import os
import subprocess
from pathlib import Path

import pytest

from tests.nix_fixtures import zephyr_fixtures_root
from tests.toolchains import host_cc

FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures"


FIXTURE_VARIANTS: dict[str, tuple[str, tuple[str, ...]]] = {
	"pie": ("device_model.c", ("-g", "-O0")),
	"nopie": ("device_model.c", ("-g", "-O0", "-no-pie")),
	"emit_relocs": ("device_model.c", ("-g", "-O0", "-no-pie", "-Wl,--emit-relocs")),
	"nodebug": ("device_model.c", ("-O0", "-no-pie")),
	"o2": ("device_model.c", ("-g", "-O2", "-no-pie")),
	"dwarf4": ("device_model.c", ("-g", "-gdwarf-4", "-O0", "-no-pie")),
	"minimal.pie": ("minimal.c", ("-g", "-O0")),
	"null": (
		"null_fn.c",
		("-g", "-O0", "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections", "-no-pie"),
	),
	"arrays": ("array_fn.c", ("-g", "-O0", "-no-pie")),
	"anonymous": ("anonymous_types.c", ("-g", "-O0", "-no-pie")),
	"residue": ("residue.c", ("-g", "-O0", "-no-pie")),
	"threads": ("static_threads.c", ("-g", "-O0", "-no-pie")),
	"shared_threads": ("shared_threads.c", ("-g", "-O0", "-no-pie")),
}


def _compile(source: Path, output: Path, *flags: str) -> Path:
	subprocess.run(
		[*host_cc(*flags), str(source), "-o", str(output)],
		check=True,
		capture_output=True,
	)
	return output


def _compile_object(source: Path, output: Path) -> Path:
	subprocess.run(
		[*host_cc("-c", "-g", "-O0"), str(source), "-o", str(output)],
		check=True,
		capture_output=True,
	)
	return output


def _compile_tus(
	source_directory: Path,
	build_directory: Path,
	files: tuple[tuple[str, tuple[str, ...]], ...],
	output_name: str,
) -> Path:
	objects: list[Path] = []
	for name, flags in files:
		output = build_directory / f"{name}.o"
		subprocess.run(
			[
				*host_cc("-c", "-g", "-O0", *flags),
				str(source_directory / f"{name}.c"),
				"-o",
				str(output),
			],
			check=True,
			capture_output=True,
		)
		objects.append(output)
	output = build_directory / f"{output_name}.elf"
	subprocess.run(
		[*host_cc("-no-pie"), *objects, "-o", str(output)], check=True, capture_output=True
	)
	return output


def _compile_multi(source_directory: Path, build_directory: Path) -> Path:
	definition_object = build_directory / "multi_def.o"
	use_object = build_directory / "multi_use.o"
	output = build_directory / "multi.elf"
	subprocess.run(
		[
			*host_cc("-c", "-O0"),
			str(source_directory / "multi_def.c"),
			"-o",
			str(definition_object),
		],
		check=True,
		capture_output=True,
	)
	subprocess.run(
		[
			*host_cc("-c", "-g", "-O0"),
			str(source_directory / "multi_use.c"),
			"-o",
			str(use_object),
		],
		check=True,
		capture_output=True,
	)
	subprocess.run(
		[*host_cc("-no-pie"), str(definition_object), str(use_object), "-o", str(output)],
		check=True,
		capture_output=True,
	)
	return output


@pytest.fixture(scope="session")
def fixture_elfs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
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
	variants["object"] = _compile_object(
		FIXTURE_DIRECTORY / "minimal.c", build_directory / "minimal.o"
	)
	variants["decl"] = _compile_tus(
		FIXTURE_DIRECTORY,
		build_directory,
		(("decl_def", ()), ("decl_use", ("-DSECOND",)), ("decl_fwd", ())),
		"decl",
	)
	return variants


@pytest.fixture(scope="session")
def zephyr_fixtures() -> Path:
	return zephyr_fixtures_root(os.environ)
