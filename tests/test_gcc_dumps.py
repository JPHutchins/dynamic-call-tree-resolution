# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Descriptor records from the alias dumps a build's own GCC wrote, as the plugin writes them (#205),
and what every store writes (#341)."""

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
from dynamic_call_tree_resolution.descriptors import load_descriptors, parse_descriptors
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
from dynamic_call_tree_resolution.gcc_dumps.taken import taken_line, unit_taken
from dynamic_call_tree_resolution.gcc_dumps.units import Record, UnitTypes, unit_types
from dynamic_call_tree_resolution.gcc_dumps.writes import EVERY_STORE, unit_writes, write_line
from dynamic_call_tree_resolution.model import (
	EmbeddedStructMember,
	FunctionPointerMember,
	MemberExtent,
	StructureLayout,
)
from tests.toolchains import FIXTURES, cortex_m3_unit, host_cc

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
			"site\tthrough_array_pointer\t160\t10\tfield\tstruct api\trun",
			"site\tthrough_cast\t90\t10\tother\tint (*<Txxxx>) (int)",
			"site\tthrough_deref\t94\t10\tother\tint (*<Txxxx>) (int)",
			"site\tthrough_direct_suffix\t183\t20\tparameter\tdev",
			"site\tthrough_element\t52\t9\tarray\ttable",
			"site\tthrough_field\t40\t9\tfield\tstruct api\trun",
			"site\tthrough_local_array\t138\t9\tfield\tstruct api\trun",
			"site\tthrough_local_static\t150\t9\tvariable\tsaved.1",
			"site\tthrough_member_table\t130\t9\tarray\t<expression>",
			"site\tthrough_nested\t56\t9\tfield\tstruct <anonymous>\tfn",
			"site\tthrough_parameter\t44\t9\tparameter\tcb",
			"site\tthrough_return_slot\t193\t20\tparameter\tmake",
			"site\tthrough_returned_slot\t198\t9\tparameter\tmake",
			"site\tthrough_stored_result\t202\t27\tparameter\tmake",
			"site\tthrough_struct_element\t98\t9\tfield\tstruct api\trun",
			"site\tthrough_suffixed\t174\t24\tvariable\thook_1",
			"site\tthrough_suffixed\t174\t9\tfield\tstruct api\trun",
			"site\tthrough_unsized_element\t102\t9\tfield\tstruct api\trun",
			"site\tthrough_variable\t48\t9\tvariable\thook",
			"site\tthrough_volatile\t64\t2\tfield\tstruct boot\tfn",
			"store\tarray_member_store\t126\t21\tfield\tstruct api\trun\tfunction\tput_char",
			"store\tcall_stores\t178\t11\tfield\tstruct api\trun\tother\tint (*<Txxxx>) (int)",
			"store\tcall_stores\t179\t9\tvariable\thook_1\tother\tint (*<Txxxx>) (int)",
			"store\tdata_stores\t107\t19\tfield\tstruct <anonymous>\tfn\tfunction\tput_char",
			"store\tfield_stores\t81\t19\tfield\tstruct <anonymous>\tfn\tparameter\tfn",
			"store\tfield_stores\t82\t16\tfield\tunion <anonymous>\tfirst\tnull\t0",
			"store\tinstall\t72\t7\tvariable\thook\tparameter\tfn",
			"store\tstatic_record_store\t155\t16\tfield\tstruct api\trun\tfunction\tput_char",
			"store\tsuffixed_stores\t169\t12\tfield\tstruct api\trun\tfunction\tput_char",
			"store\tsuffixed_stores\t170\t9\tvariable\thook_1\tfunction\tput_char",
			"store\tthrough_local_array\t136\t17\tfield\tstruct api\trun\tfunction\tput_char",
			"store\tthrough_local_static\t149\t8\tvariable\tsaved.1\tparameter\tnext",
			"store\ttypedef_store\t76\t9\tfield\tstruct stream\tput\tfunction\tput_char",
		)
	)
)
"""What the descriptors plugin, on Arm GNU 14.3.1, writes for gcc_dumps.c at -O2."""
_WRITES_UNIT: Final = "CMakeFiles/app.dir/gcc_dump_writes.c"
_WRITES_SOURCE: Final = "tests/fixtures/gcc_dump_writes.c"
_WRITE_RECORDS: Final = tuple(
	"\t".join((kind, _WRITES_UNIT, function, _WRITES_SOURCE, *rest))
	for kind, function, *rest in (
		line.split("\t")
		for line in (
			"pass\tassembly\t102\t2\t<asm>\t0\trecord\tstruct dev",
			"pass\tfills\t75\t2\tmemcpy\t0\tvariable\tconfig",
			"pass\tfills\t75\t2\tmemcpy\t1\tnone",
			"pass\tfills\t76\t2\tmemset\t0\trecord\tstruct dev",
			"pass\thanded\t87\t2\tsink\t0\trecord\tstruct dev",
			"pass\thanded\t87\t2\tsink\t1\tvariable\tcounter",
			"pass\thanded\t88\t2\t<indirect>\t0\trecord\tstruct dev",
			"pass\tlocal_handed\t107\t2\ttake_pair\t0\trecord\tstruct pair",
			"write\tanonymous_store\t50\t10\tfield\tstruct node\tnext",
			"write\tassembly\t101\t2\tfield\tstruct dev\tcount",
			"write\tatomics\t81\t2\tfield\tstruct dev\tcount",
			"write\tatomics\t82\t2\tvariable\tword",
			"write\tatomics\t83\t10\tvariable\tcounter",
			"write\telement_stores\t66\t11\tvariable\ttable",
			"write\telement_stores\t67\t13\tfield\tstruct dev\tregs",
			"write\tfield_store\t36\t11\tfield\tstruct dev\tcount",
			"write\tfills\t75\t2\tvariable\tconfig",
			"write\tfills\t76\t2\trecord\tstruct dev",
			"write\tlocal_handed\t106\t14\tfield\tstruct pair\ta",
			"write\tlocal_handed\t106\t14\tfield\tstruct pair\tb",
			"write\tmake_pair\t97\t9\tfield\tstruct pair\ta",
			"write\tmake_pair\t97\t9\tfield\tstruct pair\tb",
			"write\toffset_store\t62\t39\tfield\tstruct dev\tflags",
			"write\tout_parameter\t58\t7\tnone",
			"write\tpointer_to_pointer\t71\t6\tnone",
			"write\trecord_copy\t54\t5\trecord\tstruct dev",
			"write\tresult_store\t92\t11\tfield\tstruct dev\tcount",
			"write\tstatic_store\t45\t8\tvariable\tcalls.0",
			"write\tvariable_store\t40\t10\tvariable\tcounter",
		)
	)
)
"""What every store of gcc_dump_writes.c at -O2 writes, checked by hand against its source: no plugin
writes these."""
_TAKEN_RECORDS: Final = (
	"taken\tfield\tstruct dev\tcount",
	"taken\tvariable\tcounter",
	"taken\tvariable\tword",
)
"""The scalar fields and variables gcc_dump_writes.c takes the address of: ``&config`` and ``&local``
are records, and ``&expected`` is a local."""


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
	described = dump_descriptors(_dumped(tmp_path / "build"), REPOSITORY_ROOT)
	assert (described.lines, described.gaps, described.units, described.dumps) == (
		_PLUGIN_RECORDS,
		(),
		1,
		1,
	)


def test_every_store_of_a_build_is_recorded_by_what_it_writes(tmp_path: Path) -> None:
	assert dump_descriptors(
		_dumped(tmp_path / "build", FIXTURES / "gcc_dump_writes.c"), REPOSITORY_ROOT
	).writes == (*_WRITE_RECORDS, *_TAKEN_RECORDS, EVERY_STORE)


def test_only_a_whole_file_that_marks_it_records_every_store() -> None:
	records = (*_PLUGIN_RECORDS, *_WRITE_RECORDS, *_TAKEN_RECORDS)
	marked = parse_descriptors("".join(f"{line}\n" for line in (*records, EVERY_STORE)))
	assert (
		parse_descriptors("".join(f"{line}\n" for line in _PLUGIN_RECORDS)).every_store,
		parse_descriptors("".join(f"{line}\n" for line in records)).every_store,
		(marked.every_store, len(marked.writes), len(marked.passes), len(marked.taken)),
	) == (False, False, (True, 21, 8, 3))


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


def test_an_object_with_rela_relocations_names_its_types_through_them(tmp_path: Path) -> None:
	subprocess.run(
		host_cc("-c", "-O2", "-g", str(FIXTURES / "gcc_dumps.c"), "-o", str(tmp_path / "x.o")),
		check=True,
	)
	types = unit_types(tmp_path / "x.o")
	assert (
		{"put_char", "install"} <= types.functions,
		{"struct api", "struct driver"} <= types.layouts.keys(),
	) == (True, True)


def test_the_command_writes_where_narrow_by_field_reads(
	tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	build = _dumped(tmp_path / "build", FIXTURES / "gcc_dump_writes.c")
	(build / "zephyr").mkdir()
	descriptors(build, root=REPOSITORY_ROOT)
	written = load_descriptors(build / "zephyr" / "descriptors.txt")
	assert (
		len(written.sites),
		len(written.writes) + len(written.passes),
		written.every_store,
		capsys.readouterr().err,
	) == (
		1,
		len(_WRITE_RECORDS),
		True,
		(
			f"1 records and {len(_WRITE_RECORDS) + len(_TAKEN_RECORDS)} writes from the alias dumps"
			" of 1 of 1 C units\n"
		),
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
		writes=(),
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


_CALLBACK_LAYOUT: Final = StructureLayout(
	members=(FunctionPointerMember(kind="function_pointer", name="cb", offset=0, signature=None),),
	size=4,
	extents={"cb": MemberExtent(offset=0, size=4, layout=None)},
)
_TAGGED_TYPES: Final = UnitTypes(
	layouts={"struct foo": _CALLBACK_LAYOUT, "struct bar": _CALLBACK_LAYOUT},
	typedefs={"foo": Record(name="struct foo_impl", key="struct foo_impl")},
	variables={},
	functions=frozenset({"g"}),
	statics={},
)
"""``struct foo`` is both a tag and a typedef of another record, which a dump prints alike."""


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
			_function(
				"int i",
				"  int (*<T1>) (int) _2;\n  int _3;",
				"_2 = MEM <outer_t[2]> [(void *)&o][i_4(D)].inner",
				"_3 = _2 (0)",
			),
			("site\tu\tf\ta.c\t1\t1\tfield\tstruct outer\tinner",),
			(),
			id="an element of an array MEM is of the array's type, not the address's cast",
		),
		pytest.param(
			_TAGGED_TYPES,
			_function("struct foo * f", "", "f_2(D)->cb = g"),
			(),
			("f_2(D)->cb = g",),
			id="a store into a tag a typedef of another record shares a name with",
		),
		pytest.param(
			_TAGGED_TYPES,
			";; Function f (f, funcdef_no=0)\n\nvoid f (struct bar * b)\n{\n\n  <bb 2> :\n"
			"  b_1(D)->cb = g;\n}\n",
			("store\tu\tf\t<unknown>\t0\t0\tfield\tstruct bar\tcb\tfunction\tg",),
			(),
			id="a store GCC gave no location",
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


_DEVICE_LAYOUT: Final = StructureLayout(
	members=(),
	size=8,
	extents={
		"count": MemberExtent(offset=0, size=4, layout=None),
		"tag": MemberExtent(offset=4, size=2, layout=None),
		"flags": MemberExtent(offset=6, size=2, layout=None),
	},
)
_WRITE_TYPES: Final = UnitTypes(
	layouts={
		"struct dev": _DEVICE_LAYOUT,
		"struct outer": StructureLayout(
			members=(),
			size=8,
			extents={
				"head": MemberExtent(offset=0, size=4, layout=None),
				"inner": MemberExtent(offset=4, size=4, layout="struct in"),
			},
		),
		"struct in": StructureLayout(
			members=(), size=4, extents={"lo": MemberExtent(offset=0, size=2, layout=None)}
		),
		"struct empty": StructureLayout(members=(), size=0),
		"struct msg": StructureLayout(
			members=(),
			size=4,
			extents={
				"hdr": MemberExtent(offset=0, size=4, layout=None),
				"data": MemberExtent(offset=4, size=None, layout=None),
			},
		),
	},
	typedefs={},
	variables={"counter": "int"},
	functions=frozenset({"g"}),
	statics={},
)


@pytest.mark.parametrize(
	("dump", "writes"),
	[
		pytest.param(
			_function(
				"struct dev * p, struct msg * m",
				"",
				"MEM[(short int *)p_1(D) + 4294967292B] = 0",
				"MEM[(int *)m_2(D) + 4B] = 0",
				"MEM[(int *)m_2(D) + 12B] = 0",
			),
			(
				"write\tu\tf\ta.c\t0\t1\tunknown",
				"write\tu\tf\ta.c\t1\t1\tfield\tstruct msg\tdata",
				"write\tu\tf\ta.c\t2\t1\tfield\tstruct msg\tdata",
			),
			id="a negative offset, as CONTAINER_OF makes, and a flexible array past the end",
		),
		pytest.param(
			_function(
				"struct outer * o",
				"",
				"MEM[(short int *)o_1(D) + 4B] = 0",
				"MEM[(short int *)o_1(D) + 6B] = 0",
				"MEM[(int *)o_1(D) + 12B] = 0",
			),
			(
				"write\tu\tf\ta.c\t0\t1\tfield\tstruct in\tlo",
				"write\tu\tf\ta.c\t1\t1\tfield\tstruct outer\tinner",
				"write\tu\tf\ta.c\t2\t1\tunknown",
			),
			id="an offset into a nested record, at a member, between them, or past the end",
		),
		pytest.param(
			_function(
				"struct dev * d, struct unknown * u, struct empty * e, int * x, char * c",
				"",
				"MEM[(char *)d_1(D) + 3B] = 0",
				"MEM[(int *)u_2(D) + 4B] = 0",
				"MEM[(int *)e_3(D)] = 0",
				"MEM[(int *)x_4(D)] = 0",
				"MEM[(int *)&counter + 4B] = 0",
				"MEM[(int *)d_1(D) + 4B] = 0",
				"MEM[(uint16_t *)d_1(D) + 4B] = 0",
				"MEM <unsigned char[8]> [(char * {ref-all})d_1(D)] = 0",
				"MEM <struct dev> [(char *)c_5(D)] = 0",
			),
			(
				"write\tu\tf\ta.c\t0\t1\tfield\tstruct dev\tcount",
				"write\tu\tf\ta.c\t1\t1\trecord\tstruct unknown",
				"write\tu\tf\ta.c\t2\t1\trecord\tstruct empty",
				"write\tu\tf\ta.c\t3\t1\tnone",
				"write\tu\tf\ta.c\t4\t1\tvariable\tcounter",
				"write\tu\tf\ta.c\t5\t1\trecord\tstruct dev",
				"write\tu\tf\ta.c\t6\t1\trecord\tstruct dev",
				"write\tu\tf\ta.c\t7\t1\trecord\tstruct dev",
				"write\tu\tf\ta.c\t8\t1\trecord\tstruct dev",
			),
			id="offsets a layout can't place, accesses wider or less known than a member",
		),
		pytest.param(
			_function(
				"struct dev[2] * rows, int[4] * cells",
				"  int _3;",
				"(*rows_1(D))[_3] = 0",
				"(*cells_2(D))[_3] = 0",
			),
			(
				"write\tu\tf\ta.c\t0\t1\trecord\tstruct dev",
				"write\tu\tf\ta.c\t1\t1\tnone",
			),
			id="an element through a pointer to an array",
		),
		pytest.param(
			_function(
				"int n, struct dev s",
				"  struct dev * p;\n  int _3;\n  static int state_1;",
				"n = 1",
				"s = {}",
				"p = _3",
				"undeclared->count = 0",
				"REALPART_EXPR <c> = 0",
				"undeclared.count = 0",
				"state_1 = 1",
			),
			(
				"write\tu\tf\ta.c\t0\t1\tnone",
				"write\tu\tf\ta.c\t1\t1\trecord\tstruct dev",
				"write\tu\tf\ta.c\t2\t1\tnone",
				"write\tu\tf\ta.c\t3\t1\tnone",
				"write\tu\tf\ta.c\t4\t1\tunknown",
				"write\tu\tf\ta.c\t5\t1\tvariable\tundeclared",
				"write\tu\tf\ta.c\t6\t1\tvariable\tstate_1",
			),
			id="parameters, a local pointer, a member of nothing declared, a complex part",
		),
		pytest.param(
			_function(
				"int * q",
				"  int * _2;\n  int _3;\n  int _4;",
				"_2 = &counter",
				"__atomic_store_4 (_2, 1, 5)",
				"__sync_synchronize ()",
				"_3 = __atomic_load_4 (&counter, 5)",
				"_4 = __atomic_compare_exchange_4 (&counter, q_1(D), 1, 0, 5, 5)",
				"counter = g ()",
				"if (_3 != 0)",
				"undeclared_5 = {CLOBBER(eos)}",
				"sink (q_1(D), _3, unknown_9)",
				'__builtin_puts (&"exit"[0])',
				'sink ("(%d, %s", &counter)',
			),
			(
				"write\tu\tf\ta.c\t1\t1\tvariable\tcounter",
				"write\tu\tf\ta.c\t4\t1\tvariable\tcounter",
				"write\tu\tf\ta.c\t4\t1\tnone",
				"write\tu\tf\ta.c\t5\t1\tvariable\tcounter",
				"pass\tu\tf\ta.c\t8\t1\tsink\t0\tnone",
				"write\tu\tf\ta.c\t9\t1\tnone",
				"pass\tu\tf\ta.c\t10\t1\tsink\t1\tvariable\tcounter",
			),
			id="builtins, a call's result, a condition, undumped code and a string",
		),
		pytest.param(
			_function(
				"struct dev * d",
				"",
				'__asm__ goto("b %l0" :  :  :  : lab)',
				'__asm__ __volatile__("x" :  : weird)',
				'__asm__ __volatile__("str %1, %0" : "=m" counter : "r" d_1(D))',
				'__asm__ __volatile__("nop")',
				'__asm__ __volatile__("mrs %0, basepri" : "=r" counter)',
				'__asm__ __volatile__("dsb" :  :  : "memory")',
				'__asm__ __volatile__("str r0, [r1]" :  :  : "memory")',
			),
			(
				"write\tu\tf\ta.c\t0\t1\tunknown",
				"write\tu\tf\ta.c\t1\t1\tunknown",
				"write\tu\tf\ta.c\t2\t1\tvariable\tcounter",
				"pass\tu\tf\ta.c\t2\t1\t<asm>\t1\trecord\tstruct dev",
				"write\tu\tf\ta.c\t4\t1\tvariable\tcounter",
				"write\tu\tf\ta.c\t6\t1\tunknown",
			),
			id="inline assembly: unreadable, outputs, and a memory clobber that may or can't store",
		),
	],
)
def test_a_write_names_what_its_store_writes(dump: str, writes: tuple[str, ...]) -> None:
	assert (
		tuple(
			write_line(record)
			for record in unit_writes(
				"u", dump_functions(dump), _WRITE_TYPES, REPOSITORY_ROOT, frozenset({"f", "g"})
			)
		)
		== writes
	)


def test_only_scalar_fields_and_variables_whose_address_is_used_are_taken() -> None:
	dump = (
		";; Function f (f, funcdef_no=0)\n\nint f (struct dev * d, struct outer * o)\n{\n"
		"  int * _2;\n  struct dev local;\n\n"
		"  # _6 = PHI <&counter(2), 0B(3)>\n"
		"  [a.c:1:1] _2 = &d_1(D)->count;\n"
		"  [a.c:2:1] sink (&o_3(D)->inner, &local, &o_3(D)->head);\n"
		"  [a.c:3:1] _4 = &table_end - &table_start;\n"
		"  [a.c:4:1] if (&flag != 0B)\n"
		'  [a.c:5:1] __asm__ __volatile__("" :  : "r" &word);\n'
		"  [a.c:6:1] return &d_1(D)->undescribed;\n"
		"  [a.c:7:1] _5 = &REALPART_EXPR <c>;\n"
		"}\n"
	)
	assert tuple(
		taken_line(target) for target in unit_taken(dump_functions(dump), _WRITE_TYPES)
	) == (
		"taken\tfield\tstruct dev\tcount",
		"taken\tfield\tstruct outer\thead",
		"taken\tvariable\ttable_end",
		"taken\tvariable\ttable_start",
		"taken\tvariable\tflag",
		"taken\tvariable\tword",
		"taken\tfield\tstruct dev\tundescribed",
		"taken\tunknown",
		"taken\tvariable\tcounter",
	)


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
