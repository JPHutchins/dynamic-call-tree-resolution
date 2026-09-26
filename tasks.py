# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Task definitions for dynamic-call-tree-resolution."""

from pathlib import Path

from camas import Claude, Config, Parallel, Sequential, Task

format = Task("uv run ruff format {paths}", mutates=True, paths=".")
format_check = Task("uv run ruff format --check {paths}", paths=".")
lint = Task("uv run ruff check {paths}", paths=".")
lint_fix = Task("uv run ruff check --fix {paths}", mutates=True, paths=".")
fix = Sequential(lint_fix, format)
actionlint = Task("uv run actionlint")
mypy = Task("uv run mypy .")
pyright = Task("uv run pyright src tests")
ty = Task("uv run ty check")
zuban = Task("uv run zuban check src tests")
pyrefly = Task("uv run pyrefly check")

typecheck = Parallel(mypy, pyright, ty, zuban, pyrefly)
test = Task("uv run pytest --doctest-modules -v -m 'not slow'")
coverage = Task(
	"uv run pytest --doctest-modules -m 'not slow' --cov --cov-report=term-missing --cov-report=xml"
)

all = Sequential(fix, Parallel(actionlint, typecheck, coverage))
check = Parallel(format_check, lint, actionlint, typecheck, test)
gate = Parallel(format_check, lint, actionlint, typecheck, coverage)

matrix = Sequential(
	Task("uv sync"),
	check,
	env={"UV_PROJECT_ENVIRONMENT": ".camas/.venv-{PY}", "UV_PYTHON": "{PY}"},
	matrix={
		"PY": tuple(
			stripped
			for line in (Path(__file__).parent / ".python-version").read_text().splitlines()
			if (stripped := line.strip()) and not stripped.startswith("#")
		)
	},
)

_ = Config(default_task=all, github_task=check, agent=Claude(fix=fix, check=gate))
