# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""RTOS models, and the thread entries they start with known arguments."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
from salix import Struct, replace

from dynamic_call_tree_resolution import (
	Address,
	LinkReference,
	Machine,
	ReferenceKind,
	build_report,
	load,
	resolve,
)
from dynamic_call_tree_resolution.cli import analyze, stack
from dynamic_call_tree_resolution.model import (
	BARE_METAL,
	ArmCore,
	ArmProfile,
	Dead,
	Dispatch,
	HandlerStack,
	Interrupt,
	InterruptStack,
	ModeStack,
	PrivilegedStack,
	RtosModel,
	StructureLayout,
	SystemThread,
	ThreadCreation,
	Uncounted,
)
from dynamic_call_tree_resolution.points_to import pointer_at
from dynamic_call_tree_resolution.rtos import RtosChoice, rtos_model, zephyr
from dynamic_call_tree_resolution.vsa.fallback import referenced_only_at, referrers
from tests.programs import build_program
from tests.toolchains import (
	GdbStop,
	ZephyrMachine,
	zephyr_console_cortex_m3,
	zephyr_stack_pointers,
	zephyr_values,
)

if TYPE_CHECKING:
	from collections.abc import Callable

	from dynamic_call_tree_resolution.model import Program

PACKAGE: Final = Path(__file__).parent.parent / "src" / "dynamic_call_tree_resolution"

ADAPTERS: Final = "dynamic_call_tree_resolution.rtos"


def _imports(node: ast.AST) -> tuple[str, ...]:
	match node:
		case ast.Import(names=names):
			return tuple(alias.name for alias in names)
		case ast.ImportFrom(module=str() as module):
			return (module,)
		case _:
			return ()


def _imported(path: Path) -> frozenset[str]:
	return frozenset(
		name for node in ast.walk(ast.parse(path.read_text())) for name in _imports(node)
	)


def _imports_an_adapter(path: Path) -> bool:
	return any(name == ADAPTERS or name.startswith(f"{ADAPTERS}.") for name in _imported(path))


def test_only_the_cli_imports_an_rtos_adapter() -> None:
	assert {
		path.relative_to(PACKAGE).parts[0]
		for path in PACKAGE.rglob("*.py")
		if "rtos" not in path.relative_to(PACKAGE).parts and _imports_an_adapter(path)
	} == {"cli"}


def test_an_image_without_zephyr_is_bare_metal(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["nopie"])
	assert (
		zephyr.detect(program),
		rtos_model(program, RtosChoice.AUTO),
		rtos_model(program, RtosChoice.NONE),
	) == (None, BARE_METAL, BARE_METAL)


def test_zephyr_skips_a_writable_record_and_one_without_an_entry(
	fixture_elfs: dict[str, Path],
) -> None:
	program = load(fixture_elfs["threads"])
	model = rtos_model(program, RtosChoice.ZEPHYR)
	assert (
		sorted(
			(thread.name, program.functions[thread.entry].name, thread.arguments[1:])
			for thread in model.threads
		),
		resolve(program, model).seeded,
	) == (
		[("escaped_tid", "escaped_thread", (0, 0)), ("rom_tid", "rom_thread", (0, 0))],
		frozenset({"rom_tid"}),
	)


def test_cli_analyze_says_which_thread_entries_are_seeded(
	fixture_elfs: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
	analyze(fixture_elfs["threads"])
	assert {
		line for line in capsys.readouterr().out.splitlines() if line.startswith("thread ")
	} == {
		"thread escaped_tid: escaped_thread (not seeded: its address is taken elsewhere)",
		"thread rom_tid: rom_thread (seeded from its record)",
	}


@pytest.mark.parametrize(
	("slots", "seeded"),
	[
		pytest.param((0x2000,), frozenset({Address(0x1001)}), id="only in its record"),
		pytest.param((0x2000, 0x2100), frozenset[Address](), id="also outside its record"),
	],
)
def test_an_entry_the_linker_also_references_outside_its_record_is_not_seeded(
	slots: tuple[int, ...], seeded: frozenset[Address]
) -> None:
	program = replace(
		build_program(
			Machine.EM_ARM,
			(("entry", 0x1001, 2),),
			objects=(("record", 0x2000, (0x1001).to_bytes(4, "little")),),
			sections={0x1000: bytes.fromhex("00bf")},
			pointer_size=4,
		),
		link_references=tuple(
			LinkReference(
				slot=Address(slot),
				symbol="entry",
				value=Address(0x1001),
				kind=ReferenceKind.ADDRESS,
			)
			for slot in slots
		),
	)
	assert (
		referenced_only_at(
			referrers(program, frozenset({Address(0x1001)})),
			{Address(0x1001): frozenset({Address(0x2000)})},
		)
		== seeded
	)


def test_choosing_zephyr_for_an_image_without_it_is_an_error(
	fixture_elfs: dict[str, Path],
) -> None:
	with pytest.raises(ValueError, match="no Zephyr"):
		rtos_model(load(fixture_elfs["nopie"]), RtosChoice.ZEPHYR)


@pytest.mark.image
def test_zephyr_is_detected_in_hello_world_which_defines_no_static_thread(
	zephyr_fixtures: Path,
) -> None:
	assert rtos_model(
		load(zephyr_fixtures / "hello" / "zephyr" / "zephyr.elf"), RtosChoice.AUTO
	) == RtosModel(
		name="zephyr",
		evidence=("z_thread_entry", "struct _static_thread_data"),
		threads=(),
		trampoline="z_thread_entry",
		creation=ThreadCreation(
			frame_builders=("arch_new_thread", "arch_switch_to_main_thread"),
			setup="z_setup_new_thread",
			static_start="z_init_static_threads",
			static_entries=frozenset(),
		),
		stack_reservation=16,
		exception_frame=36,
		system_threads=(
			SystemThread(name="z_main_thread", entry=Address(0xCD9)),
			SystemThread(name="z_idle_threads", entry=Address(0x1E05)),
		),
		interrupt_stack=InterruptStack(name="z_interrupt_stacks"),
		dispatches=(
			Dispatch(
				function="z_sys_init_run_level",
				targets=frozenset(
					{Address(0x105), Address(0x8C9), Address(0x8ED), Address(0x91D), Address(0xAA1)}
				),
			),
			Dispatch(function="do_device_init", targets=frozenset({Address(0x1D4F)})),
		),
	)


@pytest.fixture(scope="module")
def userspace_program(zephyr_fixtures: Path) -> Program:
	return load(zephyr_fixtures / "userspace" / "zephyr" / "zephyr.elf")


@pytest.mark.image
def test_zephyrs_syscall_dispatch_goes_to_each_handler_its_table_holds(
	userspace_program: Program,
) -> None:
	model = rtos_model(userspace_program, RtosChoice.AUTO)
	(targets,) = (
		dispatch.targets for dispatch in model.dispatches if dispatch.function == "z_arm_do_syscall"
	)
	names = {userspace_program.functions[target].name for target in targets}
	assert (
		len(names),
		{"z_mrsh_k_sem_take", "z_mrsh_k_sem_give", "handler_bad_syscall"} <= names,
		[
			site.dispatch
			for site in resolve(userspace_program, model).sites
			if userspace_program.functions[site.caller_address].name == "z_arm_do_syscall"
		],
	) == (89, True, [targets])


@pytest.mark.image
def test_an_m_profile_user_thread_makes_its_system_calls_on_a_privileged_stack(
	zephyr_fixtures: Path, userspace_program: Program
) -> None:
	addresses = {
		function.name: address for address, function in userspace_program.functions.items()
	}
	assert (
		rtos_model(
			userspace_program,
			RtosChoice.AUTO,
			zephyr.kconfig((zephyr_fixtures / "userspace" / "zephyr" / ".config").read_text()),
		).privileged_stack,
		rtos_model(
			replace(
				userspace_program,
				arm_core=ArmCore(profile=ArmProfile.REAL_TIME, floating_point=False),
			),
			RtosChoice.AUTO,
		).privileged_stack,
	) == (
		PrivilegedStack(
			name="priv_stacks",
			system_call=addresses["z_arm_do_syscall"],
			user_start=addresses["arch_user_mode_enter"],
			saved=8,
			stack_size=1024,
		),
		None,
	)


@pytest.mark.image
@pytest.mark.parametrize(
	("function", "below_top"),
	[
		pytest.param("arch_user_mode_enter", 0, id="a user thread starts at the top"),
		pytest.param("z_arm_do_syscall", 8, id="a system call enters two words below it"),
	],
)
def test_where_each_path_on_a_privileged_stack_starts_on_qemu(
	zephyr_fixtures: Path, function: str, below_top: int
) -> None:
	stack_pointer, start = zephyr_values(
		ZephyrMachine.MPS2_AN385,
		zephyr_fixtures / "userspace" / "zephyr" / "zephyr.elf",
		(GdbStop(f"*{function}", ("$sp", "_kernel.cpus[0].current->arch.priv_stack_start")),),
	)
	assert start + 1024 - stack_pointer == below_top


@pytest.mark.image
def test_cli_stack_reports_the_privileged_stacks_from_the_system_call_entry(
	zephyr_fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	artifacts = zephyr_fixtures / "userspace"
	stack(artifacts, artifacts / "zephyr" / "zephyr.elf", path="priv_stacks")
	lines = capsys.readouterr().out.splitlines()
	assert (
		"stack reservation: 8 bytes, exception frame: 36 bytes, stack: 1024 bytes)" in lines[1],
		lines[2:4],
		[line.split(" = ")[0] for line in lines[-2:]],
	) == (
		True,
		[
			"priv_stacks +0 = 0 bytes",
			"z_arm_do_syscall +24 = 24 bytes via system call (measured, recursion)",
		],
		["(stack reservation) +8", "(exception frame) +36"],
	)


@pytest.mark.image
def test_the_syscall_dispatch_needs_a_read_only_table_that_holds_only_functions(
	userspace_program: Program,
) -> None:
	(table,) = (
		data_object
		for data_object in userspace_program.objects.values()
		if data_object.name == "_k_syscall_table"
	)
	section = max(start for start in userspace_program.sections if start <= table.address)
	assert [
		"z_arm_do_syscall"
		in {dispatch.function for dispatch in rtos_model(program, RtosChoice.AUTO).dispatches}
		for program in (
			replace(
				userspace_program,
				sections={
					**userspace_program.sections,
					section: replace(userspace_program.sections[section], writable=True),
				},
			),
			replace(
				userspace_program,
				functions={
					address: function
					for address, function in userspace_program.functions.items()
					if function.name != "handler_bad_syscall"
				},
			),
		)
	] == [False, False]


@pytest.mark.parametrize(
	("name", "sys_init", "device_init"),
	[
		(
			"hello",
			{
				"boot_banner",
				"malloc_prepare",
				"sys_clock_driver_init",
				"uart_console_init",
				"uart_stellaris_init",
			},
			{"uart_stellaris_init"},
		),
		(
			"sensor-threads",
			{
				"boot_banner",
				"malloc_prepare",
				"sys_clock_driver_init",
				"uart_console_init",
				"uart_stellaris_init",
			},
			{
				"adt7420_init",
				"bmi160_init",
				"i2c_emul_init",
				"spi_emul_init",
				"uart_stellaris_init",
			},
		),
	],
)
@pytest.mark.image
def test_zephyrs_init_dispatch_goes_to_the_entries_its_linker_sections_hold(
	zephyr_fixtures: Path, name: str, sys_init: set[str], device_init: set[str]
) -> None:
	program = load(zephyr_fixtures / name / "zephyr" / "zephyr.elf")
	assert {
		dispatch.function: {program.functions[target].name for target in dispatch.targets}
		for dispatch in rtos_model(program, RtosChoice.AUTO).dispatches
	} == {"z_sys_init_run_level": sys_init, "do_device_init": device_init}


@pytest.mark.image
def test_an_init_entry_outside_its_section_bounds_leaves_its_dispatch_unmodeled(
	zephyr_fixtures: Path,
) -> None:
	program = load(zephyr_fixtures / "hello" / "zephyr" / "zephyr.elf")
	assert [
		dispatch.function
		for dispatch in rtos_model(
			replace(
				program,
				labels={**program.labels, "__init_end": program.labels["__init_start"]},
			),
			RtosChoice.AUTO,
		).dispatches
	] == ["do_device_init"]


@pytest.mark.image
def test_the_main_and_idle_threads_start_their_stack_reservation_below_their_stack_tops_on_qemu(
	zephyr_fixtures: Path,
) -> None:
	program = load(zephyr_fixtures / "hello" / "zephyr" / "zephyr.elf")
	assert set(
		zephyr_stack_pointers(
			ZephyrMachine.LM3S6965EVB,
			zephyr_fixtures / "hello" / "zephyr" / "zephyr.elf",
			"*z_thread_entry",
			2,
		)
	) == {
		stack.address + stack.size - model.stack_reservation
		for model in (rtos_model(program, RtosChoice.AUTO),)
		for stack in program.objects.values()
		if stack.name in {"z_main_stack", "z_idle_stacks"}
	}


@pytest.mark.image
def test_a_system_thread_takes_the_trampolines_frame_and_calls_from_its_code_without_su_or_ci(
	zephyr_fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	build_directory = zephyr_fixtures / "hello"
	stack(build_directory, build_directory / "zephyr" / "zephyr.elf", path="z_main_thread")
	main_thread = capsys.readouterr().out.splitlines()[1:4]
	stack(build_directory, build_directory / "zephyr" / "zephyr.elf", path="z_idle_threads")
	assert (main_thread, capsys.readouterr().out.splitlines()[1:]) == (
		[
			"z_main_thread: unbounded, at least 476 bytes (recursion: 15, unmeasured: 1, measured: 45, stack reservation: 16 bytes, exception frame: 36 bytes, stack: 1024 bytes)",
			"z_main_thread +8 = 8 bytes (measured)",
			"bg_thread_main +40 = 48 bytes via system thread (measured, recursion)",
		],
		[
			"z_idle_threads: 228 bytes (measured: 5, stack reservation: 16 bytes, exception frame: 36 bytes, stack: 256 bytes, margin: 28 bytes)",
			"z_idle_threads +8 = 8 bytes (measured)",
			"z_impl_k_thread_abort +168 = 176 bytes via binary (measured)",
			"(stack reservation) +16 = 192 bytes",
			"(exception frame) +36 = 228 bytes",
		],
	)


@pytest.fixture(scope="module")
def hello_program(zephyr_fixtures: Path) -> Program:
	return load(zephyr_fixtures / "hello" / "zephyr" / "zephyr.elf")


@pytest.mark.image
@pytest.mark.parametrize(
	("config", "setup_symbol", "reservation"),
	[
		pytest.param(
			"", True, 4 + 8 + 4, id="without .config, the setup symbol and 8-byte alignment"
		),
		pytest.param("", False, 0, id="without .config or the setup symbol, none"),
		pytest.param(
			'CONFIG_ARCH="arm"\nCONFIG_THREAD_LOCAL_STORAGE=y\nCONFIG_STACK_ALIGN_DOUBLE_WORD=y\n',
			False,
			4 + 8 + 4,
			id="TLS from .config when link-time optimization inlined the setup",
		),
		pytest.param(
			'CONFIG_ARCH="arm"\nCONFIG_THREAD_LOCAL_STORAGE=y\n',
			True,
			4 + 8,
			id="4-byte alignment without CONFIG_STACK_ALIGN_DOUBLE_WORD",
		),
		pytest.param(
			'CONFIG_ARCH="arm"\nCONFIG_THREAD_LOCAL_STORAGE=y\nCONFIG_STACK_ALIGN_DOUBLE_WORD=y\n'
			"CONFIG_STACK_POINTER_RANDOM=64\n",
			True,
			4 + 8 + 63 + 5,
			id="the largest random offset",
		),
		pytest.param(
			'CONFIG_ARCH="arm"\nCONFIG_THREAD_USERSPACE_LOCAL_DATA=y\n'
			"CONFIG_STACK_ALIGN_DOUBLE_WORD=y\n",
			False,
			4 + 4,
			id="the userspace-local data, rounded",
		),
		pytest.param(
			'CONFIG_ARCH="arm"\nCONFIG_THREAD_LOCAL_STORAGE=y\nCONFIG_THREAD_USERSPACE_LOCAL_DATA=y\n',
			True,
			4 + 8 + 4,
			id="the userspace-local data below TLS",
		),
	],
)
def test_the_stack_reservation_follows_the_builds_config(
	hello_program: Program, config: str, setup_symbol: bool, reservation: int
) -> None:
	assert (
		rtos_model(
			hello_program
			if setup_symbol
			else replace(
				hello_program,
				symbol_addresses={
					name: addresses
					for name, addresses in hello_program.symbol_addresses.items()
					if name != "arch_tls_stack_setup"
				},
			),
			RtosChoice.ZEPHYR,
			zephyr.kconfig(config),
		).stack_reservation
		== reservation
	)


@pytest.mark.image
@pytest.mark.parametrize(
	("config", "not_counted"),
	[
		pytest.param(
			'CONFIG_ARCH="arm"\n',
			frozenset[Uncounted](),
			id="hello adds nothing the model leaves out",
		),
		pytest.param(
			'CONFIG_ARCH="arm"\nCONFIG_USE_SWITCH=y\n',
			frozenset({Uncounted.SWITCH_FRAME}),
			id="switch frame",
		),
	],
)
def test_zephyrs_thread_rows_leave_out_what_the_builds_config_adds_to_their_stacks(
	hello_program: Program, config: str, not_counted: frozenset[Uncounted]
) -> None:
	assert (
		rtos_model(hello_program, RtosChoice.ZEPHYR, zephyr.kconfig(config)).not_counted
		== not_counted
	)


@pytest.mark.image
@pytest.mark.parametrize(
	("config", "check_symbol", "sentinel"),
	[
		pytest.param('CONFIG_ARCH="arm"\n', False, 0, id="without the sentinel"),
		pytest.param('CONFIG_ARCH="arm"\nCONFIG_STACK_SENTINEL=y\n', False, 4, id="from .config"),
		pytest.param("", True, 4, id="without .config, from its check"),
	],
)
def test_zephyrs_thread_rows_add_the_stack_sentinel_the_build_writes(
	hello_program: Program, config: str, check_symbol: bool, sentinel: int
) -> None:
	assert (
		rtos_model(
			replace(
				hello_program,
				symbol_addresses={
					**hello_program.symbol_addresses,
					**(
						{"z_check_stack_sentinel": frozenset({Address(0x1000)})}
						if check_symbol
						else {}
					),
				},
			),
			RtosChoice.ZEPHYR,
			zephyr.kconfig(config),
		).stack_sentinel
		== sentinel
	)


@pytest.mark.image
def test_each_userspace_thread_starts_its_stack_reservation_below_its_stack_top_on_qemu(
	zephyr_fixtures: Path,
) -> None:
	elf = zephyr_fixtures / "userspace" / "zephyr" / "zephyr.elf"
	program = load(elf)
	assert set(zephyr_stack_pointers(ZephyrMachine.MPS2_AN385, elf, "*z_thread_entry", 4)) == {
		stack.address + stack.size - model.stack_reservation
		for model in (rtos_model(program, RtosChoice.AUTO),)
		for stack in program.objects.values()
		if stack.name
		in {
			"z_main_stack",
			"z_idle_stacks",
			"_k_thread_stack_user_tid",
			"_k_thread_stack_supervisor_tid",
		}
	}


@pytest.mark.image
def test_zephyr_writes_the_stack_sentinel_in_the_lowest_word_of_a_threads_stack_on_qemu(
	zephyr_fixtures: Path,
) -> None:
	assert zephyr_values(
		ZephyrMachine.MPS3_AN536,
		zephyr_fixtures / "synchronization-r52" / "zephyr" / "zephyr.elf",
		(GdbStop("thread_b_entry_point", ("*(unsigned int *)&_k_thread_stack_thread_b",)),),
	) == (0xF0F0F0F0,)


@pytest.mark.image
@pytest.mark.parametrize(
	("symbol", "not_counted"),
	[
		pytest.param("arm_m_exc_exit", frozenset({Uncounted.SWITCH_FRAME}), id="Cortex-M switch"),
		pytest.param("z_arm_context_switch", frozenset({Uncounted.SWITCH_FRAME}), id="A/R switch"),
	],
)
def test_without_a_config_a_symbol_only_its_build_defines_shows_what_zephyr_adds(
	hello_program: Program, symbol: str, not_counted: frozenset[Uncounted]
) -> None:
	assert (
		rtos_model(
			replace(
				hello_program,
				symbol_addresses={
					**hello_program.symbol_addresses,
					symbol: frozenset({Address(0x1000)}),
				},
			),
			RtosChoice.ZEPHYR,
		).not_counted
		== not_counted
	)


@pytest.mark.image
def test_without_a_config_the_userspace_local_data_type_shows_zephyr_reserves_it(
	hello_program: Program,
) -> None:
	assert (
		rtos_model(
			replace(
				hello_program,
				symbol_addresses={
					name: addresses
					for name, addresses in hello_program.symbol_addresses.items()
					if name != "arch_tls_stack_setup"
				},
				layouts={
					**hello_program.layouts,
					"struct _thread_userspace_local_data": StructureLayout(members=(), size=4),
				},
			),
			RtosChoice.ZEPHYR,
		).stack_reservation
		== 4 + 4
	)


@pytest.mark.image
@pytest.mark.parametrize(
	("machine", "core", "not_counted"),
	[
		pytest.param(
			Machine.EM_ARM,
			ArmCore(profile=ArmProfile.REAL_TIME, floating_point=True),
			frozenset[Uncounted](),
			id="R-profile, counted from its interrupt table",
		),
		pytest.param(
			Machine.EM_ARM,
			ArmCore(profile=ArmProfile.APPLICATION, floating_point=False),
			frozenset[Uncounted](),
			id="A-profile, counted from its interrupt table",
		),
		pytest.param(
			Machine.EM_ARM,
			None,
			frozenset({Uncounted.EXCEPTION_FRAMES}),
			id="no profile in the build attributes",
		),
		pytest.param(
			Machine.EM_386, None, frozenset[Uncounted](), id="x86 stays outside the model"
		),
	],
)
def test_zephyrs_thread_rows_leave_out_the_exception_frames_of_a_core_with_no_profile(
	hello_program: Program,
	machine: Machine,
	core: ArmCore | None,
	not_counted: frozenset[Uncounted],
) -> None:
	assert (
		rtos_model(
			replace(hello_program, machine=machine, arm_core=core), RtosChoice.ZEPHYR
		).not_counted
		== not_counted
	)


@pytest.mark.image
@pytest.mark.parametrize(
	("name", "not_counted"),
	[
		pytest.param("userspace", frozenset[Uncounted](), id="userspace on mps2/an385"),
		pytest.param(
			"synchronization-r52",
			frozenset[Uncounted](),
			id="synchronization with the stack sentinel on a Cortex-R52",
		),
	],
)
def test_without_its_config_a_testbed_still_shows_what_zephyr_adds_to_its_thread_stacks(
	zephyr_fixtures: Path, name: str, not_counted: frozenset[Uncounted]
) -> None:
	assert (
		rtos_model(
			load(zephyr_fixtures / name / "zephyr" / "zephyr.elf"), RtosChoice.AUTO
		).not_counted
		== not_counted
	)


@pytest.fixture(scope="module")
def r52_program(zephyr_fixtures: Path) -> Program:
	return load(zephyr_fixtures / "synchronization-r52" / "zephyr" / "zephyr.elf")


@pytest.mark.image
def test_r52s_z_cstart_never_falls_out_of_its_inlined_switch_to_main_thread(
	r52_program: Program,
) -> None:
	assert {
		f"{site.site_address:#x}": isinstance(site.target, Dead)
		for site in resolve(r52_program, rtos_model(r52_program, RtosChoice.AUTO)).sites
		if r52_program.functions[site.caller_address].name == "z_cstart"
	} == {"0x10003090": True}


@pytest.mark.image
def test_each_connected_interrupt_a_spurious_one_and_a_system_call_stack_a_frame_on_a_thread(
	zephyr_fixtures: Path, r52_program: Program
) -> None:
	assert [
		(model.exception_frame, model.exception_frames, model.not_counted)
		for model in (
			rtos_model(
				r52_program,
				RtosChoice.AUTO,
				zephyr.kconfig(
					(zephyr_fixtures / "synchronization-r52" / "zephyr" / ".config").read_text()
				),
			),
			rtos_model(r52_program, RtosChoice.AUTO),
		)
	] == [(32 + 200, 6 + 1 + 1, frozenset()), (32 + 200, 6 + 1 + 1, frozenset())]


@pytest.mark.image
@pytest.mark.parametrize(
	("config", "frame"),
	[
		pytest.param('CONFIG_ARCH="arm"\n', 32, id="no floating-point sharing"),
		pytest.param('CONFIG_ARCH="arm"\nCONFIG_FPU_SHARING=y\n', 32 + 72, id="sharing s0-s15"),
		pytest.param(
			'CONFIG_ARCH="arm"\nCONFIG_FPU_SHARING=y\nCONFIG_VFP_FEATURE_REGS_S64_D32=y\n',
			32 + 200,
			id="sharing d16-d31 too",
		),
	],
)
def test_an_r_profile_interrupt_frame_follows_the_builds_floating_point_sharing(
	r52_program: Program, config: str, frame: int
) -> None:
	assert rtos_model(r52_program, RtosChoice.AUTO, zephyr.kconfig(config)).exception_frame == frame


@pytest.mark.image
def test_an_r_profile_threads_frames_stay_not_counted_where_the_interrupts_cannot_be(
	r52_program: Program,
) -> None:
	(table,) = (
		data_object
		for data_object in r52_program.objects.values()
		if data_object.name == "_sw_isr_table"
	)
	section = max(start for start in r52_program.sections if start <= table.address)
	assert [
		(model.exception_frames, model.not_counted)
		for model in (
			rtos_model(
				r52_program,
				RtosChoice.AUTO,
				zephyr.kconfig('CONFIG_ARCH="arm"\nCONFIG_DYNAMIC_INTERRUPTS=y\n'),
			),
			rtos_model(
				replace(
					r52_program,
					objects={
						address: data_object
						for address, data_object in r52_program.objects.items()
						if data_object.name != "_sw_isr_table"
					},
				),
				RtosChoice.AUTO,
			),
			rtos_model(
				replace(
					r52_program,
					sections={
						**r52_program.sections,
						section: replace(r52_program.sections[section], writable=True),
					},
				),
				RtosChoice.AUTO,
			),
		)
	] == [(1, frozenset({Uncounted.EXCEPTION_FRAMES}))] * 3


@pytest.mark.image
def test_an_r_profile_core_runs_its_reset_faults_and_fast_interrupts_on_their_mode_stacks(
	zephyr_fixtures: Path, r52_program: Program
) -> None:
	addresses = {function.name: address for address, function in r52_program.functions.items()}
	assert rtos_model(
		r52_program,
		RtosChoice.AUTO,
		zephyr.kconfig(
			(zephyr_fixtures / "synchronization-r52" / "zephyr" / ".config").read_text()
		),
	).mode_stacks == (
		ModeStack(
			name="z_arm_sys_stack",
			entries=(addresses["z_arm_relocate_vector_table"], addresses["z_prep_c"]),
			stack_size=1024,
		),
		ModeStack(name="z_arm_fiq_stack", entries=(addresses["z_arm_nmi"],), stack_size=256),
		ModeStack(
			name="z_arm_abort_stack",
			entries=(addresses["z_arm_fault_data"], addresses["z_arm_fault_prefetch"]),
			exception_frame=32 + 200,
			stack_size=256,
		),
		ModeStack(
			name="z_arm_undef_stack",
			entries=(
				addresses["z_arm_fault_undef_instruction"],
				addresses["z_arm_fault_undef_instruction_fp"],
			),
			exception_frame=32 + 200,
			stack_size=256,
		),
	)


@pytest.mark.image
@pytest.mark.parametrize(
	("config", "core", "marked"),
	[
		pytest.param(
			'CONFIG_ARCH="arm"\nCONFIG_EXTRA_EXCEPTION_INFO=y\n',
			ArmCore(profile=ArmProfile.REAL_TIME, floating_point=True),
			["z_arm_abort_stack", "z_arm_undef_stack"],
			id="extra exception info marks the fault stacks",
		),
		pytest.param(
			'CONFIG_ARCH="arm"\nCONFIG_USE_SWITCH=y\n',
			ArmCore(profile=ArmProfile.REAL_TIME, floating_point=True),
			None,
			id="no mode stacks when the context switches on the thread's stack",
		),
		pytest.param(
			'CONFIG_ARCH="arm"\n',
			ArmCore(profile=ArmProfile.MICROCONTROLLER, floating_point=False),
			None,
			id="no mode stacks on M-profile",
		),
	],
)
def test_mode_stacks_follow_the_core_and_what_the_fault_entry_stacks(
	r52_program: Program, config: str, core: ArmCore, marked: list[str] | None
) -> None:
	stacks = rtos_model(
		replace(r52_program, arm_core=core), RtosChoice.AUTO, zephyr.kconfig(config)
	).mode_stacks
	assert ([stack.name for stack in stacks if stack.not_counted] if stacks else None) == marked


@pytest.mark.image
def test_the_reset_path_and_a_fault_start_where_the_mode_stack_rows_say_on_qemu(
	r52_program: Program, zephyr_fixtures: Path
) -> None:
	"""The fault is a prefetch abort: a jump into a stack, which the MPU keeps from executing."""
	reset, sys_stack, _, fault, abort_stack = zephyr_values(
		ZephyrMachine.MPS3_AN536,
		zephyr_fixtures / "synchronization-r52" / "zephyr" / "zephyr.elf",
		(
			GdbStop("*z_prep_c", ("$sp", "&z_arm_sys_stack")),
			GdbStop("thread_b_entry_point", ("$pc = &z_arm_fiq_stack",)),
			GdbStop("*z_arm_fault_prefetch", ("$sp", "&z_arm_abort_stack")),
		),
	)
	frames = {
		stack.name: stack.exception_frame
		for stack in rtos_model(r52_program, RtosChoice.AUTO).mode_stacks
	}
	assert (
		sys_stack + 1024 - reset,
		frames["z_arm_sys_stack"],
		abort_stack + 256 - fault,
		frames["z_arm_abort_stack"],
	) == (0, 0, 32 + 200, 32 + 200)


@pytest.mark.image
def test_cli_stack_reports_each_r_profile_mode_stack_in_place_of_its_handlers(
	zephyr_fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	artifacts = zephyr_fixtures / "synchronization-r52"
	stack(artifacts, artifacts / "zephyr" / "zephyr.elf")
	rows = {line.split(":")[0]: line for line in capsys.readouterr().out.splitlines()}
	endings = {
		"z_arm_sys_stack": "stack: 1024 bytes)",
		"z_arm_fiq_stack": "stack: 256 bytes)",
		"z_arm_abort_stack": "exception frame: 232 bytes, stack: 256 bytes)",
		"z_arm_undef_stack": "exception frame: 232 bytes, stack: 256 bytes)",
	}
	assert (
		{name: rows[name][-len(ending) :] for name, ending in endings.items()},
		[
			name
			for name in ("z_prep_c", "z_arm_nmi", "z_arm_fault_data", "z_arm_fault_prefetch")
			if name in rows
		],
	) == (endings, [])


@pytest.mark.image
def test_an_r_profile_core_runs_interrupt_handlers_and_system_calls_on_its_svc_stack(
	zephyr_fixtures: Path, r52_program: Program
) -> None:
	addresses = {function.name: address for address, function in r52_program.functions.items()}
	(gpio,) = {
		address
		for address, function in r52_program.functions.items()
		if function.name == "gpio_cmsdk_ahb_isr"
	}
	assert rtos_model(
		r52_program,
		RtosChoice.AUTO,
		zephyr.kconfig(
			(zephyr_fixtures / "synchronization-r52" / "zephyr" / ".config").read_text()
		),
	).handler_stack == HandlerStack(
		name="z_arm_svc_stack",
		interrupts=(
			Interrupt(number=18, handler=addresses["uart_cmsdk_apb_isr"]),
			Interrupt(number=27, handler=addresses["arm_arch_timer_compare_isr"]),
			*(Interrupt(number=number, handler=gpio) for number in range(58, 62)),
		),
		spurious=(Interrupt(number=0, handler=addresses["z_irq_spurious"]),),
		wrapper_calls=(addresses["arm_gic_get_active"], addresses["arm_gic_eoi"]),
		interrupt_frame=24,
		system_calls=(addresses["z_arm_do_swap"], addresses["z_do_kernel_oops"]),
		system_call_frame=16,
		stack_size=512,
	)


@pytest.mark.image
@pytest.mark.parametrize(
	("config", "interrupt_frame"),
	[
		pytest.param("", 24 + 8, id="without .config, tracking the active interrupt too"),
		pytest.param('CONFIG_ARCH="arm"\nCONFIG_ARM_TRACK_ACTIVE_IRQ=y\n', 24 + 8, id="tracked"),
		pytest.param('CONFIG_ARCH="arm"\nCONFIG_DYNAMIC_INTERRUPTS=y\n', None, id="uncountable"),
	],
)
def test_the_svc_stack_follows_the_builds_interrupt_options(
	r52_program: Program, config: str, interrupt_frame: int | None
) -> None:
	stack = rtos_model(r52_program, RtosChoice.AUTO, zephyr.kconfig(config)).handler_stack
	assert (stack.interrupt_frame if stack is not None else None) == interrupt_frame


@pytest.mark.image
def test_a_handler_and_a_system_call_start_where_the_svc_stack_row_says_on_qemu(
	r52_program: Program, zephyr_fixtures: Path
) -> None:
	swap, swap_stack, handler, handler_stack = zephyr_values(
		ZephyrMachine.MPS3_AN536,
		zephyr_fixtures / "synchronization-r52" / "zephyr" / "zephyr.elf",
		(
			GdbStop("*z_arm_do_swap", ("$sp", "&z_arm_svc_stack")),
			GdbStop("*arm_arch_timer_compare_isr", ("$sp", "&z_arm_svc_stack")),
		),
	)
	stack = rtos_model(
		r52_program,
		RtosChoice.AUTO,
		zephyr.kconfig(
			(zephyr_fixtures / "synchronization-r52" / "zephyr" / ".config").read_text()
		),
	).handler_stack
	assert stack is not None
	assert (swap_stack + 512 - swap, handler_stack + 512 - handler) == (
		stack.system_call_frame,
		stack.interrupt_frame,
	)


@pytest.mark.image
def test_cli_stack_nests_every_connected_handler_and_a_spurious_one_on_the_svc_stack(
	zephyr_fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	artifacts = zephyr_fixtures / "synchronization-r52"
	stack(artifacts, artifacts / "zephyr" / "zephyr.elf", path="z_arm_svc_stack")
	lines = capsys.readouterr().out.splitlines()
	assert (
		lines[1].endswith(
			"nested exceptions: 7, priority levels: 6 (interrupt table), exception frame: 24 "
			"bytes each, stack: 512 bytes)"
		),
		[line.split(" +")[0] for line in lines[2:]],
	) == (
		True,
		[
			"(exception 2) SVC",
			"(exception 27) arm_arch_timer_compare_isr",
			"(exception 58) gpio_cmsdk_ahb_isr",
			"(exception 59) gpio_cmsdk_ahb_isr",
			"(exception 60) gpio_cmsdk_ahb_isr",
			"(exception 61) gpio_cmsdk_ahb_isr",
			"(exception 18) uart_cmsdk_apb_isr",
			"(exception 0) z_irq_spurious",
		],
	)


@pytest.mark.image
def test_an_r_profile_interrupt_frame_is_what_the_isr_wrapper_stacks_on_qemu(
	r52_program: Program, zephyr_fixtures: Path
) -> None:
	"""The breakpoints sit after ``srsdb``/``cps`` and after the FPU reservation in Zephyr
	e560c91's ``_isr_wrapper``."""
	after_srsdb, after_reservation = zephyr_values(
		ZephyrMachine.MPS3_AN536,
		zephyr_fixtures / "synchronization-r52" / "zephyr" / "zephyr.elf",
		(GdbStop("*_isr_wrapper+12", ("$sp",)), GdbStop("*_isr_wrapper+20", ("$sp",))),
	)
	assert after_srsdb + 2 * 4 - after_reservation == (
		rtos_model(r52_program, RtosChoice.AUTO).exception_frame
	)


@pytest.mark.image
@pytest.mark.parametrize(
	("name", "marked", "rows"),
	[
		pytest.param(
			"userspace",
			[],
			[
				(
					"supervisor_tid: unbounded, at least 252 bytes (unmeasured: 2, stack "
					"reservation: 8 bytes, exception frame: 36 bytes, stack: 1024 bytes)"
				),
				(
					"user_tid: unbounded, at least 252 bytes (unmeasured: 2, stack reservation: 8 "
					"bytes, exception frame: 36 bytes, stack: 1024 bytes)"
				),
			],
			id="userspace on mps2/an385",
		),
		pytest.param(
			"synchronization-r52",
			[],
			[
				(
					"thread_b: 2120 bytes (measured: 5, stack reservation: 16 bytes, exception "
					"frames: 8, 232 bytes each, stack sentinel: 4 bytes, stack: 1024 bytes, margin: "
					"-1096 bytes)"
				),
			],
			id="synchronization with the stack sentinel on a Cortex-R52",
		),
	],
)
def test_cli_stack_marks_the_thread_rows_whose_testbed_adds_what_the_model_leaves_out(
	zephyr_fixtures: Path,
	capsys: pytest.CaptureFixture[str],
	name: str,
	marked: list[str],
	rows: list[str],
) -> None:
	artifacts = zephyr_fixtures / name
	stack(artifacts, artifacts / "zephyr" / "zephyr.elf")
	by_entry = {line.split(":")[0]: line for line in capsys.readouterr().out.splitlines()}
	assert (
		sorted(entry for entry, line in by_entry.items() if "not counted" in line),
		[by_entry[row.split(":")[0]] for row in rows],
	) == (marked, rows)


def _trampoline_targets(program: Program, model: RtosModel) -> list[str]:
	resolution = resolve(program, model)
	(site,) = (
		site
		for site in build_report(program, resolution.assignments, resolution.sites).call_sites
		if site.caller == "z_thread_entry"
	)
	return [candidate.name for candidate in site.candidates]


@pytest.mark.image
@pytest.mark.parametrize(
	("elf", "targets"),
	[
		pytest.param("hello/zephyr/zephyr.elf", ["bg_thread_main", "idle"], id="hello"),
		pytest.param(
			"sensor-two-impl/zephyr/zephyr.elf",
			["motion_thread", "thermal_thread", "bg_thread_main", "idle"],
			id="sensor-two-impl",
		),
		pytest.param(
			"sensor-threads/zephyr/zephyr.elf",
			["bg_thread_main", "sensor_thread", "idle"],
			id="sensor-threads",
		),
		pytest.param(
			"synchronization/zephyr/zephyr.elf",
			["thread_a_entry_point", "thread_b_entry_point", "bg_thread_main", "idle"],
			id="synchronization, with a k_thread_create thread",
		),
		pytest.param("counter-su/zephyr/zephyr.exe", [], id="counter-su calls it directly"),
	],
)
def test_the_call_that_starts_every_thread_goes_to_each_entry_its_creations_pass(
	zephyr_fixtures: Path, elf: str, targets: list[str]
) -> None:
	program = load(zephyr_fixtures / elf)
	assert _trampoline_targets(program, rtos_model(program, RtosChoice.AUTO)) == targets


@pytest.mark.image
def test_each_thread_qemu_runs_is_a_target_of_the_call_that_starts_every_thread(
	zephyr_fixtures: Path,
) -> None:
	elf = zephyr_fixtures / "synchronization" / "zephyr" / "zephyr.elf"
	program = load(elf)
	ran = frozenset(line.split(":")[0] for line in zephyr_console_cortex_m3(elf, 3)[1:])
	assert (
		ran,
		frozenset(f"{thread}_entry_point" for thread in ran)
		<= frozenset(_trampoline_targets(program, rtos_model(program, RtosChoice.AUTO))),
	) == (frozenset({"thread_a", "thread_b"}), True)


def _named(program: Program, name: str) -> Address:
	(address,) = (
		address for address, function in program.functions.items() if function.name == name
	)
	return address


class _Analyzed(Struct):
	"""A program, with the RTOS model it is analyzed under."""

	program: Program
	model: RtosModel


def _builder_with_two_entries(program: Program, model: RtosModel) -> _Analyzed:
	builder = program.functions[_named(program, "arch_new_thread")]
	assert builder.signature is not None
	return _Analyzed(
		replace(
			program,
			functions={
				**program.functions,
				builder.address: replace(
					builder,
					signature=replace(
						builder.signature,
						parameters=(*builder.signature.parameters, "function pointer"),
					),
				),
			},
		),
		model,
	)


def _trampoline_held_elsewhere(program: Program, model: RtosModel) -> _Analyzed:
	(record, *_) = (
		data_object.address
		for data_object in program.objects.values()
		if data_object.name.startswith("_k_thread_data_")
	)
	return _Analyzed(
		replace(
			program,
			link_references=(
				*program.link_references,
				LinkReference(
					slot=record,
					symbol="z_thread_entry",
					value=_named(program, "z_thread_entry"),
					kind=ReferenceKind.ADDRESS,
				),
			),
		),
		model,
	)


def _records_incomplete(program: Program, model: RtosModel) -> _Analyzed:
	assert model.creation is not None
	return _Analyzed(program, replace(model, creation=replace(model.creation, static_entries=None)))


@pytest.mark.image
@pytest.mark.parametrize(
	"perturbed",
	[
		pytest.param(_builder_with_two_entries, id="a frame builder with two entry parameters"),
		pytest.param(_trampoline_held_elsewhere, id="the trampoline held outside its builders"),
	],
)
def test_the_call_that_starts_every_thread_stays_unresolved_when_a_creation_gate_fails(
	zephyr_fixtures: Path,
	perturbed: Callable[[Program, RtosModel], _Analyzed],
) -> None:
	program = load(zephyr_fixtures / "sensor-threads" / "zephyr" / "zephyr.elf")
	analyzed = perturbed(program, rtos_model(program, RtosChoice.AUTO))
	assert _trampoline_targets(analyzed.program, analyzed.model) == []


@pytest.mark.image
def test_without_every_record_read_the_static_threads_start_what_the_analysis_reads(
	zephyr_fixtures: Path,
) -> None:
	program = load(zephyr_fixtures / "sensor-threads" / "zephyr" / "zephyr.elf")
	analyzed = _records_incomplete(program, rtos_model(program, RtosChoice.AUTO))
	assert _trampoline_targets(analyzed.program, analyzed.model) == [
		"bg_thread_main",
		"sensor_thread",
		"idle",
	]


@pytest.mark.image
def test_the_records_narrow_the_entries_their_static_thread_start_passes(
	zephyr_fixtures: Path,
) -> None:
	program = load(zephyr_fixtures / "sensor-threads" / "zephyr" / "zephyr.elf")
	model = rtos_model(program, RtosChoice.AUTO)
	assert model.creation is not None
	assert _trampoline_targets(
		program,
		replace(model, creation=replace(model.creation, static_entries=frozenset[Address]())),
	) == ["bg_thread_main", "idle"]


@pytest.mark.image
def test_a_static_thread_start_compiled_out_of_line_scopes_the_records_by_its_own_span(
	zephyr_fixtures: Path,
) -> None:
	program = load(zephyr_fixtures / "sensor-threads" / "zephyr" / "zephyr.elf")
	model = rtos_model(program, RtosChoice.AUTO)
	assert model.creation is not None
	assert _trampoline_targets(
		replace(program, inlined={}),
		replace(model, creation=replace(model.creation, static_start="bg_thread_main")),
	) == ["bg_thread_main", "sensor_thread", "idle"]


@pytest.fixture(scope="module")
def sensor_program(zephyr_fixtures: Path) -> Program:
	return load(zephyr_fixtures / "sensor-two-impl" / "zephyr" / "zephyr.elf")


@pytest.mark.image
def test_zephyr_reads_each_static_thread_from_its_record(sensor_program: Program) -> None:
	model = rtos_model(sensor_program, RtosChoice.ZEPHYR)
	assert [
		(
			thread.name,
			sensor_program.functions[thread.entry].name,
			pointer_at(sensor_program, thread.entry_slot) == thread.entry,
			thread.arguments,
		)
		for thread in model.threads
	] == [
		("motion_tid", "motion_thread", True, (0, 0, 0)),
		("thermal_tid", "thermal_thread", True, (0, 0, 0)),
	]


@pytest.mark.image
def test_each_sensor_thread_entry_starts_from_its_record(sensor_program: Program) -> None:
	assert resolve(sensor_program, rtos_model(sensor_program, RtosChoice.AUTO)).seeded == frozenset(
		{
			"motion_tid",
			"thermal_tid",
		}
	)


@pytest.mark.image
def test_cli_analyze_names_the_rtos_and_its_threads_unless_told_none(
	zephyr_fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	elf = zephyr_fixtures / "sensor-two-impl" / "zephyr" / "zephyr.elf"
	analyze(elf)
	detected = capsys.readouterr().out.splitlines()
	analyze(elf, rtos=RtosChoice.NONE)
	assert (
		detected[:3],
		[
			line
			for line in capsys.readouterr().out.splitlines()
			if line.startswith(("rtos:", "thread "))
		],
	) == (
		[
			"rtos: zephyr (detected: z_thread_entry, struct _static_thread_data)",
			"thread motion_tid: motion_thread (seeded from its record)",
			"thread thermal_tid: thermal_thread (seeded from its record)",
		],
		[],
	)


@pytest.mark.image
def test_cli_stack_reports_each_sensor_thread_in_place_of_its_entry(
	zephyr_fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	artifacts = zephyr_fixtures / "sensor-two-impl"
	stack(artifacts, artifacts / "zephyr" / "zephyr.elf")
	entries = {line.split(":")[0] for line in capsys.readouterr().out.splitlines()[1:]}
	stack(artifacts, artifacts / "zephyr" / "zephyr.elf", path="thermal_tid")
	steps = capsys.readouterr().out.splitlines()[2:4]
	stack(artifacts, artifacts / "zephyr" / "zephyr.elf", rtos=RtosChoice.NONE)
	bare_metal = {line.split(":")[0] for line in capsys.readouterr().out.splitlines()[1:]}
	threads_and_entries = {"motion_tid", "thermal_tid", "motion_thread", "thermal_thread"}
	assert (
		sorted(entries & threads_and_entries),
		steps,
		sorted(bare_metal & threads_and_entries),
	) == (
		["motion_tid", "thermal_tid"],
		["thermal_tid +8 = 8 bytes", "thermal_thread +8 = 16 bytes via thread record"],
		["motion_thread", "thermal_thread"],
	)
