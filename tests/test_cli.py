# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Tests for :mod:`dynamic_call_tree_resolution.cli`."""

import re
import sys
from typing import TYPE_CHECKING

import msgspec
import pytest
from elftools.elf.enums import ENUM_E_MACHINE

from dynamic_call_tree_resolution import (
	AddressTakenReport,
	AnalysisReport,
	AnalysisSummary,
	ComparisonReport,
	EdgeKind,
	PathStepReport,
	Reason,
	StackEntryReport,
	StackPathReport,
	UnboundedStack,
)
from dynamic_call_tree_resolution.cli import analyze, compare, main, referrers, stack, summary
from tests.dctr import dctr
from tests.expected import EXPECTED_PATHS

if TYPE_CHECKING:
	from pathlib import Path


def test_cli_analyze_json(fixture_elfs: dict[str, Path]) -> None:
	report = msgspec.json.decode(
		dctr("analyze", "--json", str(fixture_elfs["nopie"])), type=AnalysisReport
	)
	assert report.resolved_slots == 11
	assert {assignment.member_path for assignment in report.assignments} == EXPECTED_PATHS


def test_cli_analyze_plain_text(
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	analyze(fixture_elfs["nopie"])
	output = capsys.readouterr().out
	assert "dev_a.api.open: driver_a_open" in output
	assert "bss_cb: <unresolved>" in output
	assert "main@0x" in output


def test_cli_referrers_json_names_the_holder_of_each_address_taken_functions_slots(
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	referrers(fixture_elfs["nopie"], json=True)
	report = msgspec.json.decode(capsys.readouterr().out, type=tuple[AddressTakenReport, ...])
	assert {row.name: tuple(referrer.holder for referrer in row.referrers) for row in report} == {
		"__do_global_dtors_aux": ("__do_global_dtors_aux_fini_array_entry",),
		"_fini": (None,),
		"_init": (None,),
		"anon_fn": ("holder2",),
		"dev_init": ("dev_a", "dev_b", "dev_c"),
		"driver_a_close": ("ops_a",),
		"driver_a_open": ("ops_a",),
		"driver_b_close": ("ops_b",),
		"driver_b_open": ("ops_b",),
		"frame_dummy": ("__frame_dummy_init_array_entry",),
		"main": ("_start",),
		"node_fn": ("node_a",),
		"plain_target": ("plain_cb",),
		"undef_ptr_target": ("holder",),
	}


def test_cli_referrers_prints_each_slot_after_its_holder_or_alone(
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	referrers(fixture_elfs["nopie"])
	assert [
		re.sub(r"0x[0-9a-f]+", "0x…", line)
		for line in capsys.readouterr().out.splitlines()
		if line.startswith(("dev_init: ", "_fini: "))
	] == ["_fini: 0x…", "dev_init: dev_a@0x…, dev_b@0x…, dev_c@0x…"]


def test_cli_compare_json(fixture_elfs: dict[str, Path]) -> None:
	comparisons = msgspec.json.decode(
		dctr("compare", "--json", str(fixture_elfs["nopie"])), type=list[ComparisonReport]
	)
	assert [comparison.elf for comparison in comparisons] == ["device_model.nopie.elf"]
	assert comparisons[0].call_sites == 7


def test_cli_compare_json_in_process(
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	compare([fixture_elfs["nopie"]], json=True)
	comparisons = msgspec.json.decode(capsys.readouterr().out, type=list[ComparisonReport])
	assert comparisons[0].call_sites == 7


def test_cli_compare_plain_text_and_directories(
	tmp_path: Path,
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	directory = tmp_path / "elfs"
	directory.mkdir()
	(directory / "a.elf").write_bytes(fixture_elfs["nopie"].read_bytes())
	(directory / "b.elf").write_bytes(fixture_elfs["nopie"].read_bytes())
	compare([directory])
	output = capsys.readouterr().out
	assert output.count("EM_X86_64") == 2
	assert "11/2/13" in output
	assert "5/5/7" in output


def test_cli_main_entry(
	fixture_elfs: dict[str, Path],
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	monkeypatch.setattr(sys, "argv", ["dctr", "analyze", "--json", str(fixture_elfs["nopie"])])
	with pytest.raises(SystemExit):
		main()
	report = msgspec.json.decode(capsys.readouterr().out, type=AnalysisReport)
	assert report.resolved_slots == 11


def test_cli_analyze_labels_read_only_residue(
	fixture_elfs: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
	analyze(fixture_elfs["residue"])
	assert {
		"rom_ops.stop: <null>",
		"rom_arm.run: <not a function>",
		"ram_ops.stop: <unresolved>",
		"bss_ops.run: <unresolved>",
	} <= set(capsys.readouterr().out.splitlines())


def test_cli_analyze_marks_what_it_does_not_enumerate(
	fixture_elfs: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
	analyze(fixture_elfs["arrays"])
	assert "tailed_bus.tail: <not enumerated: array of unknown size>" in (
		capsys.readouterr().out.splitlines()
	)


def test_cli_stack_plain_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
	build_directory = tmp_path / "build"
	nested = build_directory / "zephyr" / "CMakeFiles" / "zephyr.dir"
	nested.mkdir(parents=True)
	(nested / "main.c.ci").write_text(
		'graph: { edge: { sourcename: "main" targetname: "worker" } }\n'
	)
	(nested / "main.c.su").write_text("main.c:1:1:main\t16\tstatic\n")
	(nested / "worker.c.su").write_text("worker.c:2:1:worker\t32\tstatic\n")
	stack(build_directory)
	assert "main: 48 bytes" in capsys.readouterr().out


def test_cli_stack_with_elf_expands_indirect_sites(
	tmp_path: Path,
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	build_directory = tmp_path / "build"
	nested = build_directory / "zephyr" / "CMakeFiles" / "zephyr.dir"
	nested.mkdir(parents=True)
	(nested / "main.c.ci").write_text(
		'graph: { edge: { sourcename: "main" targetname: "__indirect_call" } }\n'
	)
	(nested / "main.c.su").write_text("main.c:1:1:main\t16\tstatic\n")
	(nested / "plain.c.su").write_text("plain.c:2:1:plain_target\t64\tstatic\n")
	stack(build_directory, elf=fixture_elfs["nopie"])
	output = capsys.readouterr().out
	assert output.splitlines() == [
		"resolved slots: 11 | indirect call sites: 1 | not in the image: 0 | membership: names",
		"main: unbounded, at least 80 bytes (recursion: 1, unmeasured: 12)",
		"plain_target: 64 bytes",
	]


def test_cli_stack_header_says_when_sites_are_narrowed_by_signature(
	tmp_path: Path,
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	build_directory = tmp_path / "build"
	build_directory.mkdir()
	(build_directory / "main.c.ci").write_text(
		'graph: { edge: { sourcename: "main" targetname: "__indirect_call" } }\n'
	)
	(build_directory / "main.c.su").write_text("main.c:1:1:main\t16\tstatic\n")
	stack(build_directory, elf=fixture_elfs["nopie"], narrow_by_signature=True)
	assert capsys.readouterr().out.splitlines()[0] == (
		"resolved slots: 11 | indirect call sites: 1 | not in the image: 0 | membership: names"
		" | narrowed by signature"
	)


def test_cli_stack_with_elf_drops_entries_the_linker_discarded(
	tmp_path: Path,
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	build_directory = tmp_path / "build"
	build_directory.mkdir()
	(build_directory / "main.c.ci").write_text(
		'graph: { edge: { sourcename: "main" targetname: "plain_target" } }\n'
	)
	(build_directory / "main.c.su").write_text(
		"main.c:1:1:main\t16\tstatic\n"
		"main.c:2:1:plain_target\t64\tstatic\n"
		"main.c:3:1:discarded_helper\t512\tstatic\n"
	)
	stack(build_directory, fixture_elfs["nopie"])
	assert capsys.readouterr().out.splitlines() == [
		"resolved slots: 11 | indirect call sites: 0 | not in the image: 1 | membership: names",
		"main: 80 bytes",
	]


def test_cli_stack_with_elf_makes_an_entry_of_what_only_discarded_functions_call(
	tmp_path: Path,
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	build_directory = tmp_path / "build"
	build_directory.mkdir()
	(build_directory / "main.c.ci").write_text(
		'graph: { edge: { sourcename: "discarded_helper" targetname: "plain_target" } }\n'
	)
	(build_directory / "main.c.su").write_text(
		"main.c:2:1:plain_target\t64\tstatic\nmain.c:3:1:discarded_helper\t512\tstatic\n"
	)
	stack(build_directory, fixture_elfs["nopie"])
	assert capsys.readouterr().out.splitlines() == [
		"resolved slots: 11 | indirect call sites: 0 | not in the image: 1 | membership: names",
		"plain_target: 64 bytes",
	]


def test_cli_stack_with_elf_makes_each_static_thread_an_entry(
	tmp_path: Path,
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	build_directory = tmp_path / "build"
	build_directory.mkdir()
	(build_directory / "threads.c.ci").write_text(
		"graph: { "
		'edge: { sourcename: "z_thread_entry" targetname: "__indirect_call" } '
		'edge: { sourcename: "z_thread_entry" targetname: "abort_thread" } '
		"}\n"
	)
	(build_directory / "threads.c.su").write_text(
		"threads.c:8:6:z_thread_entry\t16\tstatic\n"
		"threads.c:17:6:rom_thread\t32\tstatic\n"
		"threads.c:23:6:escaped_thread\t48\tstatic\n"
		"threads.c:30:6:abort_thread\t24\tstatic\n"
	)
	stack(build_directory, fixture_elfs["threads"])
	rows = capsys.readouterr().out.splitlines()
	assert (
		[row for row in rows if row.split(":")[0] in {"rom_tid", "escaped_tid"}],
		[row for row in rows if row.split(":")[0] in {"rom_thread", "escaped_thread"}],
	) == (["escaped_tid: 64 bytes", "rom_tid: 48 bytes"], [])


def test_cli_stack_with_elf_bounds_each_thread_of_a_shared_entry_by_its_own_record(
	tmp_path: Path,
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	build_directory = tmp_path / "build"
	build_directory.mkdir()
	(build_directory / "threads.c.ci").write_text(
		"graph: { "
		'edge: { sourcename: "z_thread_entry" targetname: "__indirect_call" } '
		'edge: { sourcename: "sensor_thread" targetname: "__indirect_call" } '
		"}\n"
	)
	(build_directory / "threads.c.su").write_text(
		"threads.c:19:6:z_thread_entry\t16\tstatic\n"
		"threads.c:28:6:sensor_thread\t32\tstatic\n"
		"threads.c:12:6:run_small\t8\tstatic\n"
		"threads.c:14:6:run_big\t64\tstatic\n"
	)
	stack(build_directory, fixture_elfs["shared_threads"])
	rows = capsys.readouterr().out.splitlines()
	stack(build_directory, fixture_elfs["shared_threads"], path="small_tid")
	assert (
		[row for row in rows if row.split(":")[0] in {"small_tid", "big_tid"}],
		capsys.readouterr().out.splitlines()[2:],
	) == (
		["big_tid: 112 bytes", "small_tid: 56 bytes"],
		[
			"small_tid +16 = 16 bytes",
			"sensor_thread +32 = 48 bytes via thread record",
			"run_small +8 = 56 bytes via indirect: candidate",
		],
	)


def test_cli_stack_json_carries_every_name(
	tmp_path: Path,
	capsys: pytest.CaptureFixture[str],
) -> None:
	build_directory = tmp_path / "build"
	build_directory.mkdir()
	(build_directory / "main.c.ci").write_text(
		'graph: { edge: { sourcename: "main" targetname: "__indirect_call" } }\n'
	)
	(build_directory / "main.c.su").write_text("main.c:1:1:main\t16\tstatic\n")
	stack(build_directory, json=True)
	assert msgspec.json.decode(capsys.readouterr().out, type=tuple[StackEntryReport, ...]) == (
		StackEntryReport(
			entry="main",
			bound=UnboundedStack(
				at_least_bytes=16, recursion=(), unmeasured=(), dynamic=(), unresolved=("main",)
			),
		),
	)


def _path_build(tmp_path: Path) -> Path:
	build_directory = tmp_path / "build"
	build_directory.mkdir()
	(build_directory / "main.c.ci").write_text(
		'graph: { edge: { sourcename: "main" targetname: "worker" } '
		'edge: { sourcename: "main" targetname: "__indirect_call" } }\n'
	)
	(build_directory / "main.c.su").write_text("main.c:1:1:main\t16\tstatic\n")
	(build_directory / "worker.c.su").write_text("worker.c:2:1:worker\t32\tdynamic\n")
	return build_directory


def test_cli_stack_path_prints_the_entry_and_its_deepest_path(
	tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	stack(_path_build(tmp_path), path="main")
	assert capsys.readouterr().out.splitlines() == [
		"main: unbounded, at least 48 bytes (dynamic: 1, unresolved: 1)",
		"main +16 = 16 bytes (unresolved)",
		"worker +32 = 48 bytes via static (dynamic)",
	]


def test_cli_stack_path_json_carries_every_step(
	tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	stack(_path_build(tmp_path), path="main", json=True)
	assert msgspec.json.decode(capsys.readouterr().out, type=StackPathReport) == StackPathReport(
		entry="main",
		bound=UnboundedStack(
			at_least_bytes=48,
			recursion=(),
			unmeasured=(),
			dynamic=("worker",),
			unresolved=("main",),
		),
		path=(
			PathStepReport(
				function="main",
				frame_bytes=16,
				cumulative_bytes=16,
				edge=(),
				flags=(Reason.UNRESOLVED,),
			),
			PathStepReport(
				function="worker",
				frame_bytes=32,
				cumulative_bytes=48,
				edge=(EdgeKind.STATIC,),
				flags=(Reason.DYNAMIC,),
			),
		),
	)


def test_cli_stack_path_of_an_unknown_entry_is_an_error(tmp_path: Path) -> None:
	with pytest.raises(ValueError, match="nope is not a stack entry"):
		stack(_path_build(tmp_path), path="nope")


def test_cli_stack_path_with_elf_names_the_indirect_edge(
	tmp_path: Path,
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	build_directory = tmp_path / "build"
	build_directory.mkdir()
	(build_directory / "main.c.ci").write_text(
		'graph: { edge: { sourcename: "main" targetname: "__indirect_call" } }\n'
	)
	(build_directory / "main.c.su").write_text("main.c:1:1:main\t16\tstatic\n")
	(build_directory / "plain.c.su").write_text("plain.c:2:1:plain_target\t64\tstatic\n")
	stack(build_directory, elf=fixture_elfs["nopie"], path="main")
	assert capsys.readouterr().out.splitlines() == [
		"resolved slots: 11 | indirect call sites: 1 | not in the image: 0 | membership: names",
		"main: unbounded, at least 80 bytes (recursion: 1, unmeasured: 12)",
		"main +16 = 16 bytes (recursion)",
		"plain_target +64 = 80 bytes via indirect: candidate",
	]


def test_cli_stack_without_an_elf_reports_every_indirect_call_as_unresolved(
	tmp_path: Path,
	capsys: pytest.CaptureFixture[str],
) -> None:
	build_directory = tmp_path / "build"
	nested = build_directory / "zephyr" / "CMakeFiles" / "zephyr.dir"
	nested.mkdir(parents=True)
	(nested / "main.c.ci").write_text(
		'graph: { edge: { sourcename: "main" targetname: "__indirect_call" } }\n'
	)
	(nested / "main.c.su").write_text("main.c:1:1:main\t16\tstatic\n")
	stack(build_directory)
	assert capsys.readouterr().out.splitlines() == [
		"main: unbounded, at least 16 bytes (unresolved: 1)",
	]


def test_cli_bad_input_prints_one_line(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr(sys, "argv", ["dctr", "analyze", str(tmp_path / "missing.elf")])
	with pytest.raises(SystemExit) as error:
		main()
	assert str(error.value).startswith("dctr: ")


def test_cli_compare_warns_on_empty_directory(
	tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	empty = tmp_path / "empty"
	empty.mkdir()
	compare([empty])
	assert f"warning: {empty}: no .elf files" in capsys.readouterr().err


def test_cli_compare_skips_oversized_elfs(
	tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	directory = tmp_path / "elftest"
	directory.mkdir()
	oversized = directory / "oversized.elf"
	with oversized.open("wb") as stream:
		stream.truncate(51 * 1024 * 1024)
	compare([directory])
	assert "too large" in capsys.readouterr().err


def test_cli_compare_skips_unsupported_machines_and_rolls_up_the_rest(
	tmp_path: Path,
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	directory = tmp_path / "elfs"
	directory.mkdir()
	image = fixture_elfs["nopie"].read_bytes()
	riscv = directory / "a.elf"
	riscv.write_bytes(image[:18] + ENUM_E_MACHINE["EM_RISCV"].to_bytes(2, "little") + image[20:])
	(directory / "b.elf").write_bytes(image)
	compare([directory])
	captured = capsys.readouterr()
	assert f"warning: skipping {riscv} (EM_RISCV; unsupported machine)" in captured.err
	assert [line.split()[0] for line in captured.out.splitlines()[1:]] == ["b.elf"]


def test_cli_stack_accepts_positional_elf(
	tmp_path: Path,
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	build_directory = tmp_path / "build"
	nested = build_directory / "zephyr" / "CMakeFiles" / "zephyr.dir"
	nested.mkdir(parents=True)
	(nested / "main.c.ci").write_text(
		'graph: { edge: { sourcename: "main" targetname: "__indirect_call" } }\n'
	)
	(nested / "main.c.su").write_text("main.c:1:1:main\t16\tstatic\n")
	(nested / "plain.c.su").write_text("plain.c:2:1:plain_target\t64\tstatic\n")
	stack(build_directory, fixture_elfs["nopie"])
	assert "main: unbounded, at least 80 bytes (recursion: 1, unmeasured: 12)" in (
		capsys.readouterr().out
	)


def test_cli_summary_expands_an_unresolved_site_to_every_address_taken_function(
	tmp_path: Path,
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	build_directory = tmp_path / "build"
	nested = build_directory / "zephyr" / "CMakeFiles" / "zephyr.dir"
	nested.mkdir(parents=True)
	(nested / "main.c.ci").write_text(
		'graph: { edge: { sourcename: "main" targetname: "__indirect_call" } }\n'
	)
	(nested / "main.c.su").write_text("main.c:1:1:main\t16\tstatic\n")
	summary(build_directory, fixture_elfs["null"])
	report = msgspec.json.decode(capsys.readouterr().out, type=AnalysisSummary)
	assert (report.entry_points, report.discarded_entry_points) == (1, 0)
	assert isinstance(report.worst_case, UnboundedStack)
	assert (report.worst_case.recursion, report.worst_case.unresolved) == (("main",), ())


def test_cli_summary_json(
	tmp_path: Path,
	fixture_elfs: dict[str, Path],
	capsys: pytest.CaptureFixture[str],
) -> None:
	build_directory = tmp_path / "build"
	nested = build_directory / "zephyr" / "CMakeFiles" / "zephyr.dir"
	nested.mkdir(parents=True)
	(nested / "main.c.ci").write_text(
		'graph: { edge: { sourcename: "main" targetname: "__indirect_call" } }\n'
	)
	(nested / "main.c.su").write_text("main.c:1:1:main\t16\tstatic\n")
	summary(build_directory, fixture_elfs["nopie"])
	report = msgspec.json.decode(capsys.readouterr().out, type=AnalysisSummary)
	assert report.resolved_slots == 11
	assert report.total_slots == 13
	assert report.unresolved_slots == 2
	assert report.indirect_call_sites == 1
	assert report.worst_case_entry == "main"
	assert isinstance(report.worst_case, UnboundedStack)
	assert report.worst_case.at_least_bytes == 16
