# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Descriptor records from the alias dumps a build's own GCC wrote, as the plugin writes them (#205)."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from importlib import import_module
from pathlib import Path
from typing import Final

import msgspec
import pytest
from salix import Struct

from dynamic_call_tree_resolution.cli import descriptors
from dynamic_call_tree_resolution.gcc_dumps import DUMP_FLAG, DumpDescriptors, dump_descriptors
from dynamic_call_tree_resolution.gcc_dumps.dump import dump_functions
from dynamic_call_tree_resolution.gcc_dumps.records import (
	Gap,
	Located,
	SiteRecord,
	StoreRecord,
	record_line,
	unit_records,
)
from dynamic_call_tree_resolution.gcc_dumps.units import Record, UnitTypes
from dynamic_call_tree_resolution.model import (
	EmbeddedStructMember,
	FunctionPointerMember,
	MemberExtent,
	StructureLayout,
)
from tests.toolchains import FIXTURES, cortex_m3_unit

REPOSITORY_ROOT: Final = FIXTURES.parent.parent
_UNIT: Final = "CMakeFiles/app.dir/gcc_dumps.c"
_SOURCE: Final = "tests/fixtures/gcc_dumps.c"
_PLUGIN_RECORDS: Final = tuple(
	"\t".join((kind, _UNIT, function, _SOURCE, *rest))
	for kind, function, *rest in (
		line.split("\t")
		for line in (
			"site\tthrough_address_taken\t143\t9\tother\tint (*<Txxxx>) (int)",
			"site\tthrough_anonymous\t60\t9\tfield\tunion <anonymous>\tfirst",
			"site\tthrough_array_member\t122\t9\tfield\tstruct api\trun",
			"site\tthrough_cast\t90\t10\tother\tint (*<Txxxx>) (int)",
			"site\tthrough_deref\t94\t10\tother\tint (*<Txxxx>) (int)",
			"site\tthrough_element\t52\t9\tarray\ttable",
			"site\tthrough_field\t40\t9\tfield\tstruct api\trun",
			"site\tthrough_local_array\t138\t9\tfield\tstruct api\trun",
			"site\tthrough_local_static\t150\t9\tvariable\tsaved.1",
			"site\tthrough_member_table\t130\t9\tarray\t<expression>",
			"site\tthrough_nested\t56\t9\tfield\tstruct <anonymous>\tfn",
			"site\tthrough_parameter\t44\t9\tparameter\tcb",
			"site\tthrough_struct_element\t98\t9\tfield\tstruct api\trun",
			"site\tthrough_unsized_element\t102\t9\tfield\tstruct api\trun",
			"site\tthrough_variable\t48\t9\tvariable\thook",
			"site\tthrough_volatile\t64\t2\tfield\tstruct boot\tfn",
			"store\tarray_member_store\t126\t21\tfield\tstruct api\trun\tfunction\tput_char",
			"store\tdata_stores\t107\t19\tfield\tstruct <anonymous>\tfn\tfunction\tput_char",
			"store\tfield_stores\t81\t19\tfield\tstruct <anonymous>\tfn\tparameter\tfn",
			"store\tfield_stores\t82\t16\tfield\tunion <anonymous>\tfirst\tnull\t0",
			"store\tinstall\t72\t7\tvariable\thook\tparameter\tfn",
			"store\tstatic_record_store\t155\t16\tfield\tstruct api\trun\tfunction\tput_char",
			"store\tthrough_local_array\t136\t17\tfield\tstruct api\trun\tfunction\tput_char",
			"store\tthrough_local_static\t149\t8\tvariable\tsaved.1\tparameter\tnext",
			"store\ttypedef_store\t76\t9\tfield\tstruct stream\tput\tfunction\tput_char",
		)
	)
)
"""What the descriptors plugin, on Arm GNU 14.3.1, writes for gcc_dumps.c at -O2."""


class _Compile(Struct):
	"""One unit's compile, as compile_commands.json records it."""

	source: Path
	output: str
	command: tuple[str, ...]


def _build(directory: Path, sources: tuple[Path, ...], *flags: str) -> Path:
	"""A build directory: each source's object, its alias dump beside it, compile_commands.json."""
	(directory / "CMakeFiles" / "app.dir").mkdir(parents=True, exist_ok=True)
	compiles = tuple(
		_Compile(
			source=source,
			output=f"CMakeFiles/app.dir/{source.name}.obj",
			command=cortex_m3_unit(
				*flags, str(source), "-o", f"CMakeFiles/app.dir/{source.name}.obj"
			),
		)
		for source in sources
	)
	for compile_ in compiles:
		subprocess.run(compile_.command, cwd=directory, check=True, capture_output=True)
	(directory / "compile_commands.json").write_text(
		json.dumps(_compile_commands(directory, compiles))
	)
	return directory


def _compile_commands(
	directory: Path, compiles: tuple[_Compile, ...]
) -> tuple[dict[str, str], ...]:
	return tuple(
		{
			"directory": str(directory),
			"command": shlex.join(compile_.command),
			"file": str(compile_.source),
			"output": compile_.output,
		}
		for compile_ in compiles
	)


def _dumped(directory: Path, *sources: Path) -> Path:
	return _build(directory, sources or (FIXTURES / "gcc_dumps.c",), "-O2", "-g", DUMP_FLAG)


def test_a_builds_dumps_give_the_records_the_plugin_writes(tmp_path: Path) -> None:
	assert dump_descriptors(_dumped(tmp_path / "build"), REPOSITORY_ROOT) == DumpDescriptors(
		lines=_PLUGIN_RECORDS, gaps=(), units=1, dumps=1
	)


def test_a_unit_without_a_function_has_no_dump_and_no_records(tmp_path: Path) -> None:
	data = tmp_path / "data.c"
	data.write_text("int data_only;\n")
	described = dump_descriptors(
		_dumped(tmp_path / "build", FIXTURES / "gcc_dumps.c", data), REPOSITORY_ROOT
	)
	assert (described.units, described.dumps, described.lines) == (2, 1, _PLUGIN_RECORDS)


def test_a_unit_compiled_without_the_dump_flag_fails(tmp_path: Path) -> None:
	build = _build(tmp_path / "build", (FIXTURES / "gcc_dumps.c",), "-O2", "-g")
	with pytest.raises(ValueError, match=f"1 of 1 C units were compiled without {DUMP_FLAG}"):
		dump_descriptors(build, REPOSITORY_ROOT)


def test_an_object_without_debug_information_fails(tmp_path: Path) -> None:
	build = _build(tmp_path / "build", (FIXTURES / "gcc_dumps.c",), "-O2", DUMP_FLAG)
	with pytest.raises(ValueError, match="has no debug information: build with -g"):
		dump_descriptors(build, REPOSITORY_ROOT)


def test_a_dump_older_than_its_source_fails_and_a_missing_source_is_not_checked(
	tmp_path: Path,
) -> None:
	source = tmp_path / "gcc_dumps.c"
	source.write_text((FIXTURES / "gcc_dumps.c").read_text())
	build = _dumped(tmp_path / "build", source)
	(dump,) = build.glob("CMakeFiles/app.dir/*.alias")
	os.utime(source, (dump.stat().st_mtime + 10, dump.stat().st_mtime + 10))
	with pytest.raises(ValueError, match="1 alias dumps are older than their sources"):
		dump_descriptors(build, tmp_path)
	source.rename(tmp_path / "moved.c")
	assert len(dump_descriptors(build, tmp_path).lines) == len(_PLUGIN_RECORDS)


def test_a_function_compiled_without_optimization_has_no_dump_and_fails(tmp_path: Path) -> None:
	build = _build(tmp_path / "build", (FIXTURES / "gcc_dumps.c",), "-O0", "-g", DUMP_FLAG)
	with pytest.raises(ValueError, match="functions have no alias dump, such as "):
		dump_descriptors(build, REPOSITORY_ROOT)


def test_a_dump_of_functions_its_object_lacks_fails(tmp_path: Path) -> None:
	data = tmp_path / "data.c"
	data.write_text("int data_only;\n")
	build = _dumped(tmp_path / "build")
	subprocess.run(
		cortex_m3_unit("-O2", "-g", str(data), "-o", "CMakeFiles/app.dir/gcc_dumps.c.obj"),
		cwd=build,
		check=True,
	)
	with pytest.raises(ValueError, match="dumped functions are not in their objects"):
		dump_descriptors(build, REPOSITORY_ROOT)


def test_two_dumps_of_one_object_fail(tmp_path: Path) -> None:
	build = _dumped(tmp_path / "build")
	(dump,) = build.glob("CMakeFiles/app.dir/*.alias")
	shutil.copy(dump, dump.with_name("gcc_dumps.c.c.999t.alias"))
	with pytest.raises(
		ValueError, match=re.escape("2 alias dumps match CMakeFiles/app.dir/gcc_dumps.c.obj")
	):
		dump_descriptors(build, REPOSITORY_ROOT)


def test_a_moved_build_names_its_units_from_where_it_compiled(tmp_path: Path) -> None:
	moved = _dumped(tmp_path / "build").rename(tmp_path / "moved")
	assert dump_descriptors(moved, REPOSITORY_ROOT).lines == _PLUGIN_RECORDS


def test_a_unit_the_build_never_compiled_is_not_checked(tmp_path: Path) -> None:
	data = tmp_path / "data.c"
	data.write_text("int data_only;\n")
	build = _dumped(tmp_path / "build", FIXTURES / "gcc_dumps.c", data)
	(build / "CMakeFiles" / "app.dir" / "data.c.obj").unlink()
	described = dump_descriptors(build, REPOSITORY_ROOT)
	assert (described.units, described.dumps, described.lines) == (2, 1, _PLUGIN_RECORDS)


def test_function_scope_statics_that_share_a_name_keep_it_without_gccs_number(
	tmp_path: Path,
) -> None:
	source = tmp_path / "statics.c"
	source.write_text(
		"".join(
			f"int {name}(int x, int (*next)(int)) {{\n"
			"\tstatic int (*hook)(int);\n"
			"\tint (*const previous)(int) = hook;\n"
			"\thook = next;\n"
			"\treturn previous(x);\n"
			"}\n"
			for name in ("first", "second")
		)
	)
	assert tuple(
		line.split("\t", 6)[6]
		for line in dump_descriptors(_dumped(tmp_path / "build", source), tmp_path).lines
	) == (
		"variable\thook",
		"variable\thook",
		"variable\thook\tparameter\tnext",
		"variable\thook\tparameter\tnext",
	)


def test_the_command_writes_where_narrow_by_field_reads(
	tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	build = _dumped(tmp_path / "build")
	(build / "zephyr").mkdir()
	descriptors(build, root=REPOSITORY_ROOT)
	assert (
		(build / "zephyr" / "descriptors.txt").read_text().splitlines(),
		capsys.readouterr().err,
	) == (
		list(_PLUGIN_RECORDS),
		f"{len(_PLUGIN_RECORDS)} records from the alias dumps of 1 of 1 C units\n",
	)


def test_the_command_reports_each_store_it_cannot_place_and_writes_nothing(
	tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setattr(
		import_module("dynamic_call_tree_resolution.cli.descriptors"), "dump_descriptors", _one_gap
	)
	with pytest.raises(ValueError, match="1 stores of a function have no record"):
		descriptors(tmp_path, output=tmp_path / "out.txt")
	assert (capsys.readouterr().err, (tmp_path / "out.txt").exists()) == (
		"no record for a.c:3:4 in f: x_1(D)->fn = g\n",
		False,
	)


def _one_gap(_build_directory: Path, _root: Path) -> DumpDescriptors:
	return DumpDescriptors(
		lines=(),
		gaps=(
			Gap(
				at=Located(unit="u", function="f", file="a.c", line=3, column=4),
				statement="x_1(D)->fn = g",
			),
		),
		units=1,
		dumps=1,
	)


_NO_TYPES: Final = UnitTypes(
	layouts={}, typedefs={}, variables={}, functions=frozenset({"g"}), statics={}
)
_ANONYMOUS_TYPES: Final = UnitTypes(
	layouts={
		"struct outer": StructureLayout(
			members=(
				EmbeddedStructMember(
					kind="embedded_struct",
					name=None,
					offset=0,
					type_name="union <anonymous>@0x10",
					members=(),
				),
			),
			size=8,
			extents={"inner": MemberExtent(offset=4, size=4, layout=None)},
		),
		"union <anonymous>@0x10": StructureLayout(
			members=(
				FunctionPointerMember(kind="function_pointer", name="fn", offset=0, signature=None),
			),
			size=4,
			extents={"fn": MemberExtent(offset=0, size=4, layout=None)},
		),
		"struct <anonymous>@0x20": StructureLayout(
			members=(), size=4, extents={"fn": MemberExtent(offset=0, size=4, layout=None)}
		),
		"struct <anonymous>@0x30": StructureLayout(
			members=(), size=4, extents={"lone": MemberExtent(offset=0, size=4, layout=None)}
		),
	},
	typedefs={"outer_t": Record(name="struct outer", key="struct outer")},
	variables={"counter": "int"},
	functions=frozenset({"g"}),
	statics={},
)


def _function(parameters: str, declarations: str, *statements: str) -> str:
	return (
		f";; Function f (f, funcdef_no=0)\n\nint f ({parameters})\n{{\n{declarations}\n"
		+ "".join(f"  [a.c:{line}:1] {statement};\n" for line, statement in enumerate(statements))
		+ "}\n"
	)


@pytest.mark.parametrize(
	("types", "dump", "lines", "gaps"),
	[
		pytest.param(
			_NO_TYPES,
			_function("struct unknown * x", "  int (*<T1>) (int) _2;", "x_1(D)->fn = g"),
			(),
			("x_1(D)->fn = g",),
			id="a function stored into a record no layout describes",
		),
		pytest.param(
			_NO_TYPES,
			_function("struct unknown * x", "  int _2;", "x_1(D)->count = 0"),
			(),
			(),
			id="data stored into a record no layout describes",
		),
		pytest.param(
			_NO_TYPES,
			_function(
				"int x",
				"  int (*<T1a>) (int) _2;\n  int _3;",
				"_2 = MEM[(int *)x_1(D)].fn",
				"_3 = _2 (x_1(D))",
			),
			("site\tu\tf\ta.c\t1\t1\tother\tint (*<Txxxx>) (int)",),
			(),
			id="a call through a field of a type that is no record",
		),
		pytest.param(
			_NO_TYPES,
			_function(
				"void", "  int (*<T1>) (int) _2;\n  int _3;", "_2 = undeclared->fn", "_3 = _2 (0)"
			),
			("site\tu\tf\ta.c\t1\t1\tother\tint (*<Txxxx>) (int)",),
			(),
			id="a call through a field of a name nothing declares",
		),
		pytest.param(
			_ANONYMOUS_TYPES,
			_function(
				"outer_t * o",
				"  int (*<T1>) (int) _2;\n  int _3;",
				"_2 = o_1(D)->D.99.fn",
				"_3 = _2 (0)",
			),
			("site\tu\tf\ta.c\t1\t1\tfield\tunion <anonymous>\tfn",),
			(),
			id="an anonymous member of a typedef's record",
		),
		pytest.param(
			_ANONYMOUS_TYPES,
			_function(
				"struct unknown * o",
				"  int (*<T1>) (int) _2;\n  int _3;",
				"_2 = o_1(D)->D.99.lone",
				"_3 = _2 (0)",
			),
			("site\tu\tf\ta.c\t1\t1\tfield\tstruct <anonymous>\tlone",),
			(),
			id="an anonymous member only one anonymous record holds",
		),
		pytest.param(
			_ANONYMOUS_TYPES,
			_function(
				"struct unknown * o",
				"  int (*<T1>) (int) _2;\n  int _3;",
				"_2 = o_1(D)->D.99.fn",
				"_3 = _2 (0)",
			),
			("site\tu\tf\ta.c\t1\t1\tother\tint (*<Txxxx>) (int)",),
			(),
			id="an anonymous member two anonymous records hold",
		),
		pytest.param(
			_ANONYMOUS_TYPES,
			_function(
				"struct outer * o",
				"  int (*<T1>) (int) _2;\n  int _3;",
				"_2 = o_1(D)->inner.fn",
				"_3 = _2 (0)",
			),
			("site\tu\tf\ta.c\t1\t1\tother\tint (*<Txxxx>) (int)",),
			(),
			id="a member of a member that is no record",
		),
		pytest.param(
			_ANONYMOUS_TYPES,
			_function(
				"struct outer * o",
				"  int (*<T1>) (int) _2;\n  int _3;",
				"_2 = o_1(D)->inner[i_4(D)].fn",
				"_3 = _2 (0)",
			),
			("site\tu\tf\ta.c\t1\t1\tother\tint (*<Txxxx>) (int)",),
			(),
			id="a member of an element of a member that is no array of records",
		),
		pytest.param(
			_ANONYMOUS_TYPES,
			_function("void", "  int _2;", "_2 = counter_4 (1)", "counter = 1"),
			(),
			(),
			id="a direct call to a name like an SSA name, and an int stored",
		),
	],
)
def test_a_record_names_only_what_the_units_types_place(
	types: UnitTypes, dump: str, lines: tuple[str, ...], gaps: tuple[str, ...]
) -> None:
	records = unit_records("u", dump_functions(dump), types, REPOSITORY_ROOT)
	assert (
		tuple(
			record_line(record)
			for record in records
			if isinstance(record, SiteRecord | StoreRecord)
		),
		tuple(record.statement for record in records if isinstance(record, Gap)),
	) == (lines, gaps)


@pytest.mark.image
def test_every_arm_fixtures_dumps_give_its_plugin_descriptors(zephyr_fixtures: Path) -> None:
	builds = sorted(path.parent.parent for path in zephyr_fixtures.glob("*/zephyr/descriptors.txt"))
	assert {
		build.name: "".join(
			f"{line}\n" for line in dump_descriptors(build, _build_root(build)).lines
		)
		for build in builds
	} == {build.name: (build / "zephyr" / "descriptors.txt").read_text() for build in builds}


def _build_root(build: Path) -> Path:
	"""The directory the fixtures' replay took paths relative to: the build's ``testbeds/..``."""
	(first, *_) = msgspec.json.decode(
		(build / "compile_commands.json").read_bytes(), type=list[dict[str, str]]
	)
	return Path(re.sub(r"/testbeds/build/[^/]+$", "", first["directory"]))
