# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Task definitions for dynamic-call-tree-resolution."""

from pathlib import Path

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
	"nixfmt {paths}", mutates=True, paths=by_suffix((".nix",), default=("flake.nix",))
)
nix_format_check = Task(
	"nixfmt --check {paths}", paths=by_suffix((".nix",), default=("flake.nix",))
)
fix = Sequential(lint_fix, format, c_format, nix_format)
actionlint = Task("uv run actionlint", when=".github")
mypy = Task("uv run mypy .")
pyright = Task("uv run pyright src tests")

typecheck = Parallel(mypy, pyright)
test = Task("uv run pytest -v -m 'not slow'", agent_format=("--junitxml {report}", "junit"))
coverage = Task(
	"uv run pytest -m 'not slow' --cov --cov-report=term-missing --cov-report=xml",
	agent_format=("--junitxml {report}", "junit"),
)

all = Sequential(fix, Parallel(actionlint, typecheck, coverage))
check = Parallel(format_check, c_format_check, nix_format_check, lint, actionlint, typecheck, test)
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

zephyr_sdk = Path("/home/jp/zephyr-sdk-1.0.1")
hello = Task(
	"uv run --group build west build -b qemu_cortex_m3 -d ../.camas/build/hello zephyr/samples/hello_world",
	cwd=Path("testbeds"),
	env={
		"ZEPHYR_SDK_INSTALL_DIR": str(zephyr_sdk),
		"DTC": str(zephyr_sdk / "hosttools/sysroots/x86_64-pokysdk-linux/usr/bin/dtc"),
	},
	help="build the zephyr hello_world testbed for qemu_cortex_m3 (needs west update)",
)
counter = Task(
	"uv run --group build west build -b native_sim -d ../.camas/build/counter zephyr/samples/drivers/can/counter",
	cwd=Path("testbeds"),
	env={
		"ZEPHYR_TOOLCHAIN_VARIANT": "host",
		"DTC": str(zephyr_sdk / "hosttools/sysroots/x86_64-pokysdk-linux/usr/bin/dtc"),
	},
	help="build the zephyr CAN counter testbed for native_sim (device API indirection)",
)
counter_su = Task(
	"uv run --group build west build -b native_sim -d ../.camas/build/counter-su zephyr/samples/drivers/can/counter -- '-DEXTRA_CFLAGS=-fstack-usage -fcallgraph-info=su,da'",
	cwd=Path("testbeds"),
	env={
		"ZEPHYR_TOOLCHAIN_VARIANT": "host",
		"DTC": str(zephyr_sdk / "hosttools/sysroots/x86_64-pokysdk-linux/usr/bin/dtc"),
	},
	help="build the CAN counter testbed with -fstack-usage/-fcallgraph-info for WCS analysis",
)

zephyr_sdk_legacy = Path("/home/jp/zephyr-sdk-0.17.4")
zmk = Task(
	"uv run --group build west build -b nice_nano -d ../../.camas/build/zmk -s /home/jp/repos/dynamic-call-tree-resolution/testbeds/zmk/app -- -DSHIELD=a_dux_left",
	cwd=Path("testbeds/zmk-workspace"),
	env={
		"ZEPHYR_SDK_INSTALL_DIR": str(zephyr_sdk),
		"ZEPHYR_TOOLCHAIN_VARIANT": "zephyr",
		"DTC": str(zephyr_sdk / "hosttools/sysroots/x86_64-pokysdk-linux/usr/bin/dtc"),
	},
	help="build the ZMK testbed for nice_nano",
)
zswatch_patch = Task(
	"sed -i -e '/select DEPRECATED/d' -e 's/^    if kconf.warnings:/    if False and kconf.warnings:/' testbeds/zswatch-workspace/zephyr/subsys/usb/device/Kconfig testbeds/zswatch-workspace/zephyr/drivers/usb/device/Kconfig testbeds/zswatch-workspace/zephyr/scripts/kconfig/kconfig.py",
	mutates=True,
	help="testbed-local fork patches: drop DEPRECATED selects, make kconfig warnings non-fatal",
)
zswatch_build = Task(
	"uv run --group build west build -b zswatch_legacy/nrf5340/cpuapp -d ../../.camas/build/zswatch -s /home/jp/repos/dynamic-call-tree-resolution/testbeds/zswatch/app -- -DBOARD_ROOT=/home/jp/repos/dynamic-call-tree-resolution/testbeds/zswatch/app",
	cwd=Path("testbeds/zswatch-workspace"),
	env={
		"ZEPHYR_SDK_INSTALL_DIR": str(zephyr_sdk_legacy),
		"ZEPHYR_TOOLCHAIN_VARIANT": "zephyr",
		"DTC": str(zephyr_sdk_legacy / "sysroots/x86_64-pokysdk-linux/usr/bin/dtc"),
	},
	help="build the ZSWatch testbed (produces ipc_radio netcore ELF; app image blocked upstream)",
)
zswatch = Sequential(zswatch_patch, zswatch_build)

sensor_two_impl = Task(
	"uv run --group build west build -b qemu_cortex_m3 -d ../.camas/build/sensor-two-impl -s /home/jp/repos/dynamic-call-tree-resolution/tests/fixtures/sensor-two-impl-app -- '-DEXTRA_CFLAGS=-fstack-usage -fcallgraph-info=su,da'",
	cwd=Path("testbeds"),
	env={
		"ZEPHYR_SDK_INSTALL_DIR": str(zephyr_sdk),
		"DTC": str(zephyr_sdk / "hosttools/sysroots/x86_64-pokysdk-linux/usr/bin/dtc"),
	},
	help="build the two-impl sensor fixture (qemu_cortex_m3) with stack/callgraph artifacts",
)

pexplorer_testdata = Task(
	"uv run dctr compare references/pexplorer/testdata/elf_testdata",
	help="run dctr's resolution rollup over pexplorer's shared testdata ELFs "
	"(needs the references/pexplorer submodule and its git-lfs objects)",
)

_ = Config(default_task=all, github_task=matrix, agent=Claude(fix=fix, check=gate))
