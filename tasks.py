# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Task definitions for dynamic-call-tree-resolution."""

import json
from pathlib import Path
from typing import cast

from camas import Claude, Config, Parallel, Sequential, Task, by_glob, by_suffix

fixture_c_globs = ("tests/fixtures/**/*.c", "tests/fixtures/**/*.h")
fixture_c_scope = by_glob(
	fixture_c_globs,
	default=tuple(
		sorted(
			path.relative_to(Path(__file__).parent).as_posix()
			for glob in fixture_c_globs
			for path in Path(__file__).parent.glob(glob)
		)
	),
)

format = Task("uv run ruff format {paths}", mutates=True, paths=".")
format_check = Task("uv run ruff format --check {paths}", paths=".")
lint = Task("uv run ruff check {paths}", paths=".", agent_format=("--output-format sarif", "sarif"))
lint_fix = Task("uv run ruff check --fix {paths}", mutates=True, paths=".")
c_format = Task(
	"jphfmt -i {paths}",
	mutates=True,
	paths=fixture_c_scope,
	help="format the C fixtures with jphfmt",
)
c_format_check = Task(
	"jphfmt --check {paths}",
	paths=fixture_c_scope,
	help="the C fixtures must stay jphfmt-canonical",
)
nix_format = Task(
	"nixfmt {paths}",
	mutates=True,
	paths=by_suffix((".nix",), default=("flake.nix", "testbeds/fixtures.nix")),
)
nix_format_check = Task(
	"nixfmt --check {paths}",
	paths=by_suffix((".nix",), default=("flake.nix", "testbeds/fixtures.nix")),
)
fix = Sequential(lint_fix, format, c_format, nix_format)
actionlint = Task("uv run actionlint", when=".github")
mypy = Task("uv run mypy .")
pyright = Task("uv run pyright src tests tasks.py")

typecheck = Parallel(mypy, pyright)
test = Task("uv run pytest -v -m 'not slow'", agent_format=("--junitxml {report}", "junit"))
test_fast = Task(
	"uv run pytest -m 'not slow and not image'",
	agent_format=("--junitxml {report}", "junit"),
	help="the tests minus those analyzing a committed firmware image (marked image)",
)
coverage = Task(
	"uv run pytest -m 'not slow' --cov --cov-report=term-missing --cov-report=xml",
	agent_format=("--junitxml {report}", "junit"),
)

all = Sequential(fix, Parallel(actionlint, typecheck, coverage))
check = Parallel(format_check, c_format_check, nix_format_check, lint, actionlint, typecheck, test)
check_fast = Parallel(
	format_check,
	c_format_check,
	nix_format_check,
	lint,
	actionlint,
	typecheck,
	test_fast,
	help="check for quick iteration: every leaf of check, with test_fast in place of test",
)
gate = Parallel(
	format_check, c_format_check, nix_format_check, lint, actionlint, typecheck, coverage
)

matrix = Sequential(
	Task("uv python install --no-bin {PYTHON_DOWNLOAD}"),
	Task("uv sync"),
	check,
	env={"UV_PROJECT_ENVIRONMENT": ".camas/.venv-{PY}", "UV_PYTHON": "{PY}"},
	variants=tuple(
		{"PY": request, "PYTHON_DOWNLOAD": request.removesuffix("+gil")}
		for line in (Path(__file__).parent / ".python-version").read_text().splitlines()
		if (request := line.strip()) and not request.startswith("#")
	),
	help="check on each .python-version interpreter; uv cannot download a +gil request "
	"(astral-sh/uv#17437), so the GIL build is installed by its downloadable name first",
)

testbeds_init = Task(
	"uv run --group build west update --narrow --fetch-opt=--depth=1",
	cwd=Path("testbeds"),
	mutates=True,
	help="clone the west workspace testbeds/manifest/west.yml pins: zephyr, and the one module the "
	"testbeds link, cmsis_6",
)
testbeds_lock = Sequential(
	testbeds_init,
	Task("west2nix", cwd=Path("testbeds"), mutates=True),
	help="regenerate testbeds/west2nix.toml, the Nix-hashed lock `nix build .#fixtures` fetches "
	"from; run in `nix develop .#testbeds` after editing testbeds/manifest/west.yml",
)
testbeds = Parallel(
	Task(
		"uv run --group build west build -b {BOARD} -d ../.camas/build/{NAME} -s {SOURCE} "
		"-- '-DEXTRA_CFLAGS={EXTRA_CFLAGS}'",
		cwd=Path("testbeds"),
		env={"ZEPHYR_TOOLCHAIN_VARIANT": "{TOOLCHAIN}"},
	),
	variants=tuple(
		cast(
			"list[dict[str, str]]",
			json.loads((Path(__file__).parent / "testbeds/builds.json").read_text()),
		)
	),
	help="build the zephyr testbeds of testbeds/builds.json into .camas/build/{NAME}; run in "
	"`nix develop .#testbeds`, which provides the Zephyr SDK, dtc and the multilib host gcc "
	"native_sim links with",
)

pexplorer_testdata = Sequential(
	Task("git submodule update --init --depth 1 references/pexplorer", mutates=True),
	Task(
		"git -C references/pexplorer -c core.attributesFile=../pexplorer.gitattributes "
		"lfs pull --include testdata/elf_testdata",
		mutates=True,
	),
	Task("uv run dctr compare references/pexplorer/testdata/elf_testdata"),
	help="run dctr's resolution rollup over pexplorer's shared testdata ELFs",
)

_ = Config(default_task=all, github_task=matrix, agent=Claude(fix=fix, check=gate))  # type: ignore[misc]
