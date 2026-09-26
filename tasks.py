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
	(
		"uv",
		"run",
		"--group",
		"build",
		"west",
		"build",
		"-b",
		"native_sim",
		"-d",
		"../.camas/build/counter-su",
		"zephyr/samples/drivers/can/counter",
		"--",
		"-DEXTRA_CFLAGS=-fstack-usage -fcallgraph-info=su,da",
	),
	cwd=Path("testbeds"),
	env={
		"ZEPHYR_TOOLCHAIN_VARIANT": "host",
		"DTC": str(zephyr_sdk / "hosttools/sysroots/x86_64-pokysdk-linux/usr/bin/dtc"),
	},
	help="build the CAN counter testbed with -fstack-usage/-fcallgraph-info for WCS analysis",
)

zephyr_sdk_legacy = Path("/home/jp/zephyr-sdk-0.17.4")
zmk = Task(
	(
		"uv",
		"run",
		"--group",
		"build",
		"west",
		"build",
		"-b",
		"nice_nano",
		"-d",
		"../../.camas/build/zmk",
		"-s",
		"/home/jp/repos/dynamic-call-tree-resolution/testbeds/zmk/app",
		"--",
		"-DSHIELD=a_dux_left",
	),
	cwd=Path("testbeds/zmk-workspace"),
	env={
		"ZEPHYR_SDK_INSTALL_DIR": str(zephyr_sdk),
		"ZEPHYR_TOOLCHAIN_VARIANT": "zephyr",
		"DTC": str(zephyr_sdk / "hosttools/sysroots/x86_64-pokysdk-linux/usr/bin/dtc"),
	},
	help="build the ZMK testbed for nice_nano",
)
zswatch_patch = Task(
	(
		"sed",
		"-i",
		"-e",
		"/select DEPRECATED/d",
		"-e",
		"s/^    if kconf.warnings:/    if False and kconf.warnings:/",
		"zephyr/subsys/usb/device/Kconfig",
		"zephyr/drivers/usb/device/Kconfig",
		"zephyr/scripts/kconfig/kconfig.py",
	),
	cwd=Path("testbeds/zswatch-workspace"),
	help="testbed-local fork patches: drop DEPRECATED selects, make kconfig warnings non-fatal",
)
zswatch_build = Task(
	(
		"uv",
		"run",
		"--group",
		"build",
		"west",
		"build",
		"-b",
		"zswatch/nrf5340/cpuapp",
		"-d",
		"../../.camas/build/zswatch",
		"-s",
		"/home/jp/repos/dynamic-call-tree-resolution/testbeds/zswatch/app",
		"--",
		"-Dapp_EXTRA_CONF_FILE=/home/jp/repos/dynamic-call-tree-resolution/testbeds/configs/zswatch_app.conf",
		"-DBOARD_ROOT=/home/jp/repos/dynamic-call-tree-resolution/testbeds/zswatch/app",
	),
	cwd=Path("testbeds/zswatch-workspace"),
	env={
		"ZEPHYR_SDK_INSTALL_DIR": str(zephyr_sdk_legacy),
		"ZEPHYR_TOOLCHAIN_VARIANT": "zephyr",
		"DTC": str(zephyr_sdk_legacy / "sysroots/x86_64-pokysdk-linux/usr/bin/dtc"),
	},
	help="build the ZSWatch testbed for zswatch/nrf5340/cpuapp",
)
zswatch = Sequential(zswatch_patch, zswatch_build)

_ = Config(default_task=all, github_task=check, agent=Claude(fix=fix, check=gate))
