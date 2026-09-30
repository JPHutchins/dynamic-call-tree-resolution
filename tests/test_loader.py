# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Tests for :mod:`dynamic_call_tree_resolution.loader`."""

from __future__ import annotations

import struct
import subprocess
from collections import Counter
from itertools import accumulate
from pathlib import Path

import pytest
from elftools.common.exceptions import ELFError
from elftools.elf.elffile import ELFFile
from salix import Struct

from dynamic_call_tree_resolution import (
	Address,
	EmbeddedStructMember,
	Function,
	FunctionPointerMember,
	FunctionSignature,
	Relocation,
	StructPointerMember,
	load,
)
from dynamic_call_tree_resolution.loader import defined_function_names
from dynamic_call_tree_resolution.model import aligned


def test_load_functions_from_dwarf_and_symtab(fixture_elfs: dict[str, Path]) -> None:
	names = {function.name for function in load(fixture_elfs["pie"]).functions.values()}
	assert {"driver_a_open", "driver_b_close", "plain_target", "main"} <= names
	assert "puts" not in names


def test_load_without_dwarf(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["nodebug"])
	names = {function.name for function in program.functions.values()}
	assert "driver_a_open" in names
	assert "plain_target" in names
	assert program.layouts == {}


def test_load_layouts(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["pie"])
	assert set(program.layouts) == {
		"struct ops",
		"struct device",
		"struct device_ops",
		"struct handler_holder",
		"struct node",
		"struct container",
		"struct anon_wrapper",
		"struct embedded_holder",
		"union un",
		"struct bitpacked",
	}
	assert program.layouts["struct ops"].members == (
		FunctionPointerMember(
			kind="function_pointer",
			name="open",
			offset=0,
			signature=FunctionSignature(return_type="int", parameters=("void *", "int")),
		),
		FunctionPointerMember(
			kind="function_pointer",
			name="close",
			offset=8,
			signature=FunctionSignature(return_type="int", parameters=("void *",)),
		),
	)
	assert program.layouts["struct device"].members == (
		StructPointerMember(kind="struct_pointer", name="api", offset=0, pointee="struct ops"),
		StructPointerMember(kind="struct_pointer", name="context", offset=8, pointee=None),
		EmbeddedStructMember(
			kind="embedded_struct",
			name="ops",
			offset=16,
			members=(
				FunctionPointerMember(
					kind="function_pointer",
					name="init",
					offset=0,
					signature=FunctionSignature(return_type="int", parameters=()),
				),
			),
		),
	)
	assert program.layouts["struct embedded_holder"].members == (
		EmbeddedStructMember(
			kind="embedded_struct",
			name="inner",
			offset=0,
			members=(
				FunctionPointerMember(
					kind="function_pointer",
					name="fn",
					offset=8,
					signature=FunctionSignature(return_type="int", parameters=()),
				),
			),
		),
	)
	assert program.layouts["struct bitpacked"].members == ()


def test_load_declaration_types_attach_to_symtab_objects(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["multi"])
	dev_x = next(
		data_object for data_object in program.objects.values() if data_object.name == "dev_x"
	)
	assert dev_x.type_name == "struct device"


def test_load_objects(fixture_elfs: dict[str, Path]) -> None:
	program = load(fixture_elfs["pie"])
	by_name = {data_object.name: data_object for data_object in program.objects.values()}
	assert by_name["dev_a"].type_name == "struct device"
	assert by_name["dev_a"].size == 24
	assert by_name["plain_cb"].type_name == "function pointer"
	assert by_name["plain_cb"].signature == FunctionSignature(
		return_type="void", parameters=("int",)
	)
	assert by_name["bss_holder"].size == 8
	assert program.pointer_size == 8
	assert program.byte_order == "little"
	assert program.machine == "EM_X86_64"
	assert program.sections


def test_load_rejects_relocatable_objects(fixture_elfs: dict[str, Path]) -> None:
	with pytest.raises(ValueError, match="ET_REL"):
		load(fixture_elfs["object"])


def test_load_propagates_malformed_elf_errors(tmp_path: Path) -> None:
	path = tmp_path / "corrupt.elf"
	path.write_bytes(b"\x7fELF" + b"\0" * 64)
	with pytest.raises(ELFError):
		load(path)


def test_load_rejects_unknown_machines(tmp_path: Path, fixture_elfs: dict[str, Path]) -> None:
	path = tmp_path / "aarch64.elf"
	data = bytearray(fixture_elfs["nopie"].read_bytes())
	data[18:20] = (183).to_bytes(2, "little")  # EM_AARCH64
	path.write_bytes(data)
	with pytest.raises(ValueError, match="EM_AARCH64"):
		load(path)


def test_load_rel_elf_reads_in_field_addends(tmp_path: Path) -> None:
	path = tmp_path / "rel.elf"
	path.write_bytes(_rel_elf())
	program = load(path)
	assert program.relocations == (
		Relocation(slot=Address(0x1000), target=Address(0x1044), addend=0x44, type_name="R_386_32"),
		Relocation(slot=Address(0x2000), target=Address(0), addend=0, type_name="R_386_RELATIVE"),
	)
	assert program.sections[Address(0x1000)].data == b"\x44\x10\x00\x00"


_GLOBAL_FUNCTION = 0x12
_GLOBAL_OBJECT = 0x11
_SHN_ABS = 0xFFF1


class _Symbol(Struct):
	name: str
	value: int
	size: int
	info: int = _GLOBAL_FUNCTION
	section_index: int = 1


def _rel_elf(symbols: tuple[_Symbol, ...] = (_Symbol(name="fn", value=0x1000, size=4),)) -> bytes:
	shstrtab = b"\0.text\0.rel.text\0.symtab\0.shstrtab\0.strtab\0.bss\0.extra\0.rel.extra\0"
	strtab = b"\0" + b"".join(symbol.name.encode() + b"\0" for symbol in symbols)
	sections_data = (
		b"\x44\0\0\0",
		struct.pack("<II", 0x1000, (1 << 8) | 1) + struct.pack("<II", 0x2000, 8),
		b"\0" * 16
		+ b"".join(
			struct.pack(
				"<IIIBBH", name, symbol.value, symbol.size, symbol.info, 0, symbol.section_index
			)
			for symbol, name in zip(
				symbols,
				accumulate((len(symbol.name) + 1 for symbol in symbols), initial=1),
				strict=False,
			)
		),
		shstrtab,
		strtab,
		b"\0\0\0\0",
		struct.pack("<II", 0x3000, (1 << 8) | 1),
	)
	offsets: list[int] = []
	offset = 52
	for data in sections_data:
		offsets.append(offset)
		offset += len(data)
	header = struct.pack(
		"<16sHHIIIIIHHHHHH",
		b"\x7fELF\x01\x01\x01\0" + b"\0" * 8,
		2,
		3,
		1,
		0x1000,
		0,
		offset,
		0,
		52,
		0,
		0,
		40,
		9,
		4,
	)
	sections = (
		(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
		(1, 1, 0x6, 0x1000, offsets[0], 4, 0, 0, 4, 0),
		(7, 9, 0, 0, offsets[1], 16, 3, 1, 4, 8),
		(17, 2, 0, 0, offsets[2], 16 * (1 + len(symbols)), 5, 1, 4, 16),
		(25, 3, 0, 0, offsets[3], len(shstrtab), 0, 0, 1, 0),
		(34, 3, 0, 0, offsets[4], len(strtab), 0, 0, 1, 0),
		(41, 8, 0x3, 0x2000, 0, 4, 0, 0, 4, 0),
		(46, 1, 0, 0x3000, offsets[5], 4, 0, 0, 1, 0),
		(53, 9, 0, 0, offsets[6], 8, 3, 7, 4, 8),
	)
	shdrs = b"".join(struct.pack("<IIIIIIIIII", *section) for section in sections)
	return header + b"".join(sections_data) + shdrs


def test_load_skips_symtab_functions_at_address_zero_and_one(tmp_path: Path) -> None:
	path = tmp_path / "phantom.elf"
	path.write_bytes(_rel_elf((_Symbol(name="fn", value=1, size=4),)))
	program = load(path)
	assert Address(1) not in program.functions
	assert Address(0) not in program.functions


@pytest.mark.parametrize(
	"symbols",
	[
		(_Symbol(name="real", value=0x1000, size=4), _Symbol(name="alias", value=0x1000, size=0)),
		(_Symbol(name="alias", value=0x1000, size=0), _Symbol(name="real", value=0x1000, size=4)),
	],
	ids=["alias-last", "alias-first"],
)
def test_load_keeps_the_largest_same_address_function_symbol(
	tmp_path: Path, symbols: tuple[_Symbol, ...]
) -> None:
	path = tmp_path / "alias.elf"
	path.write_bytes(_rel_elf(symbols))
	assert load(path).functions[Address(0x1000)] == Function(
		name="real", address=Address(0x1000), size=4, signature=None
	)


def test_load_skips_absolute_object_symbols(tmp_path: Path) -> None:
	path = tmp_path / "absolute.elf"
	path.write_bytes(
		_rel_elf(
			(
				_Symbol(name="fn", value=0x1000, size=4),
				_Symbol(
					name="CONFIG_X",
					value=0x2000,
					size=0,
					info=_GLOBAL_OBJECT,
					section_index=_SHN_ABS,
				),
			)
		)
	)
	assert Address(0x2000) not in load(path).objects


@pytest.mark.image
@pytest.mark.parametrize(
	"elf", ["hello_zephyr_qemu_cortex_m3.elf", "sensor-two-impl/zephyr/zephyr.elf"]
)
def test_load_arm_fixture_has_no_absolute_symbol_objects(elf: str) -> None:
	program = load(Path(__file__).parent / "fixtures" / elf)
	assert [
		data_object.name
		for data_object in program.objects.values()
		if data_object.name.startswith(("CONFIG_", "___"))
	] == []


def test_defined_function_names_include_locals_and_aliases() -> None:
	names = defined_function_names(
		Path(__file__).parent / "fixtures" / "sensor-two-impl" / "zephyr" / "zephyr.elf"
	)
	assert {"ready_thread", "z_sched_ready_locked", "memcpy", "vfprintf"} <= names
	assert "__aeabi_uldivmod" not in names


def test_defined_function_names_of_a_stripped_image_are_empty(
	tmp_path: Path, fixture_elfs: dict[str, Path]
) -> None:
	stripped = tmp_path / "stripped.elf"
	subprocess.run(
		["strip", "--strip-all", "-o", str(stripped), str(fixture_elfs["nopie"])],
		check=True,
		capture_output=True,
	)
	assert defined_function_names(stripped) == frozenset()


@pytest.mark.image
@pytest.mark.parametrize(
	"elf",
	[
		"hello_zephyr_qemu_cortex_m3.elf",
		"sensor-two-impl/zephyr/zephyr.elf",
		"counter-su/zephyr/zephyr.exe",
	],
)
def test_committed_images_name_every_function_and_parameter(elf: str) -> None:
	functions = load(Path(__file__).parent / "fixtures" / elf).functions.values()
	assert [function.address for function in functions if function.name == "<anonymous>"] == []
	assert [
		function.name
		for function in functions
		if function.signature is not None and "<unknown>" in function.signature.parameters
	] == []


@pytest.mark.image
@pytest.mark.parametrize(
	"elf", ["hello_zephyr_qemu_cortex_m3.elf", "sensor-two-impl/zephyr/zephyr.elf"]
)
def test_committed_arm_images_hold_one_function_per_entry_point(elf: str) -> None:
	entry_points = Counter(
		aligned(function.address)
		for function in load(Path(__file__).parent / "fixtures" / elf).functions.values()
	)
	assert [address for address, count in entry_points.items() if count > 1] == []


@pytest.mark.image
@pytest.mark.parametrize(
	("elf", "writable", "read_only"),
	[
		("hello_zephyr_qemu_cortex_m3.elf", "datas", ("text", "rodata", "device_area")),
		("counter-su/zephyr/zephyr.exe", ".data", (".text", ".rodata", "device_area")),
		pytest.param(
			"counter-su/zephyr/zephyr.exe",
			".got.plt",
			(".init_array", ".fini_array", ".dynamic", ".got"),
			id="relro-sections-are-read-only-and-got-plt-straddles-the-end",
		),
	],
)
def test_committed_images_mark_data_writable_and_code_read_only(
	elf: str, writable: str, read_only: tuple[str, ...]
) -> None:
	path = Path(__file__).parent / "fixtures" / elf
	with path.open("rb") as stream:
		starts = {
			section.name: Address(section.header.sh_addr)
			for section in ELFFile(stream).iter_sections()
		}
	sections = load(path).sections
	assert sections[starts[writable]].writable
	assert [name for name in read_only if sections[starts[name]].writable] == []


def test_load_marks_relro_sections_read_only(fixture_elfs: dict[str, Path]) -> None:
	with fixture_elfs["nopie"].open("rb") as stream:
		starts = {
			section.name: Address(section.header.sh_addr)
			for section in ELFFile(stream).iter_sections()
		}
	sections = load(fixture_elfs["nopie"]).sections
	assert not sections[starts[".data.rel.ro"]].writable
	assert sections[starts[".data"]].writable
