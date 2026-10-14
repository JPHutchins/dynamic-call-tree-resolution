# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Which writable objects a store through a pointer the analysis can't place may write."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from dynamic_call_tree_resolution import (
	Address,
	DataObject,
	Function,
	FunctionSignature,
	Machine,
	Program,
	Section,
)
from dynamic_call_tree_resolution.vsa.cfg import control_flow_graphs
from dynamic_call_tree_resolution.vsa.escape import Escapes, escapes
from tests.programs import build_program

if TYPE_CHECKING:
	from collections.abc import Mapping

_AFTER_FIRST: Final = (
	("second", 0x2000_0004, 4, "unsigned int"),
	("third", 0x2000_0008, 8, "long long int"),
	("zeroed", 0x2000_0010, 16, "char [16]"),
)
_OBJECTS: Final = (("first", 0x2000_0000, 4, "int"), *_AFTER_FIRST)
_ARRAY: Final = tuple((name, address, size, "struct k_sem") for name, address, size, _ in _OBJECTS)
_ROUTINES: Final = (
	("other", 0x1100, "7047"),
	("memcpy", 0x1200, "7047"),
	("memset", 0x1240, "7047"),
	("early_copy", 0x1300, "fff77ebf"),
	("padded_copy", 0x1340, "00bffff75dbf"),
)
_EVERY_OBJECT: Final = ["first", "second", "third", "zeroed"]
_ALL_BUT_FIRST: Final = ["second", "third", "zeroed"]
_BASE_ONLY: Final = "014b1860704700bf00000020"
_ARGUMENT: Final = "08b5024800f07cf808bd00bf00000020"
_WALK: Final = "08b50248084400f07bf808bd00000020"
_UNKNOWN_LENGTH_COPY: Final = "08b50248024900f0fbf808bd0000002000300000"


def _program(
	code: str,
	*,
	objects: tuple[tuple[str, int, int, str | None], ...] = _OBJECTS,
	labels: Mapping[str, int] | None = None,
	data_in_code: tuple[tuple[int, int], ...] = (),
	sections: Mapping[int, bytes] | None = None,
	arm_code: tuple[tuple[int, int], ...] = (),
	writable_code: bool = False,
	signature: FunctionSignature | None = None,
	parameterless: frozenset[int] = frozenset(),
) -> Program:
	functions = (("caller", 0x1000, code), *_ROUTINES)
	return Program(
		byte_order="little",
		pointer_size=4,
		machine=Machine.EM_ARM,
		functions={
			Address(address): Function(
				name=name,
				address=Address(address),
				size=len(body) // 2,
				signature=signature if name == "caller" else None,
			)
			for name, address, body in functions
		},
		objects={
			Address(address): DataObject(
				name=name, address=Address(address), size=size, type_name=type_name, signature=None
			)
			for name, address, size, type_name in objects
		},
		layouts={},
		relocations=(),
		sections={
			**{
				Address(address): Section(
					data=bytes.fromhex(body), writable=writable_code, executable=True
				)
				for _, address, body in functions
			},
			Address(0x2000_0000): Section(
				data=bytes(16), writable=True, load_address=Address(0x3000)
			),
			Address(0x3000): Section(data=bytes(16), writable=False),
			**{
				Address(address): Section(data=data, writable=False)
				for address, data in (sections or {}).items()
			},
		},
		data_in_code=tuple((Address(low), Address(high)) for low, high in data_in_code),
		arm_code=tuple((Address(low), Address(high)) for low, high in arm_code),
		labels={name: Address(address) for name, address in (labels or {}).items()},
		symbol_addresses={name: frozenset({Address(address)}) for name, address, _ in _ROUTINES},
		nobits=((Address(0x2000_0010), Address(0x2000_0020)),),
		parameterless=frozenset(map(Address, parameterless)),
	)


def _escapes(program: Program) -> Escapes:
	return escapes(program, tuple(control_flow_graphs(program).values()))


def _unescaped(program: Program) -> list[str]:
	return sorted(program.objects[start].name for start, _ in _escapes(program).unescaped)


@pytest.mark.parametrize(
	("code", "unescaped"),
	[
		pytest.param(_BASE_ONLY, _EVERY_OBJECT, id="only the base of its own store"),
		pytest.param(_ARGUMENT, _ALL_BUT_FIRST, id="a call argument"),
		pytest.param("014800f07db800bf00000020", _ALL_BUT_FIRST, id="a tail call argument"),
		pytest.param(
			"0149024a116070470000002004000020", _ALL_BUT_FIRST, id="a value stored to another"
		),
		pytest.param("0048704700000020", _ALL_BUT_FIRST, id="a return value"),
		pytest.param("004b704700000020", _EVERY_OBJECT, id="a scratch register at a return"),
		pytest.param(
			"08b50248043000f07bf808bd00000020",
			["first", "third", "zeroed"],
			id="a constant offset into the next object",
		),
		pytest.param(_WALK, _ALL_BUT_FIRST, id="an unknown offset stays in its object"),
		pytest.param(
			"08b50348002908bf002000f079f808bd00000020",
			_ALL_BUT_FIRST,
			id="an overwrite a predicate may skip",
		),
		pytest.param(
			"08b50248002000f07bf808bd00000020", _EVERY_OBJECT, id="an overwrite before the call"
		),
		pytest.param(
			"08b540f20400c2f2000000f079f808bd",
			["first", "third", "zeroed"],
			id="movw and movt",
		),
		pytest.param("014880ed0001704700000020", _ALL_BUT_FIRST, id="an untracked store's base"),
		pytest.param("004c9f4600000020", _ALL_BUT_FIRST, id="a jump the graph can't follow"),
		pytest.param("10b5014c10bd00bf00000020", _EVERY_OBJECT, id="a register a return restores"),
		pytest.param("08b5014800df08bd00000020", _ALL_BUT_FIRST, id="a supervisor call argument"),
		pytest.param(
			"08b50348042100eb410000f079f808bd00000020",
			["first", "second", "zeroed"],
			id="a shifted constant offset",
		),
		pytest.param(
			"08b50348042100eb510000f079f808bd00000020",
			_ALL_BUT_FIRST,
			id="an offset behind a shift the scan doesn't model",
		),
		pytest.param(
			"08b5024806c800f07bf808bd00000020",
			["first", "second", "zeroed"],
			id="a load multiple's write-back",
		),
		pytest.param(
			"08b5034850f8041b00f07af808bd00bf00000020",
			["first", "third", "zeroed"],
			id="a post-indexed load's write-back",
		),
		pytest.param(
			"08b5034850f8081f00f07af808bd00bf00000020",
			["first", "second", "zeroed"],
			id="a pre-indexed load's write-back",
		),
	],
)
def test_an_object_escapes_where_its_address_leaves_the_code_that_names_it(
	code: str, unescaped: list[str]
) -> None:
	assert _unescaped(_program(code)) == unescaped


@pytest.mark.parametrize(
	("return_type", "unescaped"),
	[pytest.param("int", _ALL_BUT_FIRST, id="int"), pytest.param("void", _EVERY_OBJECT, id="void")],
)
def test_a_return_passes_registers_only_when_its_function_returns_a_value(
	return_type: str, unescaped: list[str]
) -> None:
	assert (
		_unescaped(
			_program(
				"0048704700000020",
				signature=FunctionSignature(return_type=return_type, parameters=()),
			)
		)
		== unescaped
	)


@pytest.mark.parametrize(
	("parameterless", "unescaped"),
	[
		pytest.param(frozenset[int](), _ALL_BUT_FIRST, id="a callee with parameters"),
		pytest.param(frozenset({0x1100}), _EVERY_OBJECT, id="a callee with none"),
	],
)
def test_a_call_passes_argument_registers_only_when_its_callee_takes_parameters(
	parameterless: frozenset[int], unescaped: list[str]
) -> None:
	assert _unescaped(_program(_ARGUMENT, parameterless=parameterless)) == unescaped


@pytest.mark.parametrize(
	("code", "table", "unescaped"),
	[
		pytest.param(
			"10b5044c012903d8dfe801f0010200bf10bd00bf00000020",
			(0x100C, 0x100E),
			_EVERY_OBJECT,
			id="a bounded table",
		),
		pytest.param(
			"10b5034cdfe801f0010200bf10bd00bf00000020",
			(0x1008, 0x100A),
			_ALL_BUT_FIRST,
			id="an unbounded table",
		),
	],
)
def test_a_switch_lets_every_register_escape_unless_the_graph_follows_it(
	code: str, table: tuple[int, int], unescaped: list[str]
) -> None:
	assert _unescaped(_program(code, data_in_code=(table,))) == unescaped


@pytest.mark.parametrize(
	("code", "unescaped"),
	[
		pytest.param(
			"08b503480349082200f0faf808bd00bf0000002000300000",
			_EVERY_OBJECT,
			id="a copy from the load image",
		),
		pytest.param(
			"08b503480349082200f0faf808bd00bf0000002004300000",
			["third", "zeroed"],
			id="a copy from elsewhere",
		),
		pytest.param(
			"08b503480349182200f0faf808bd00bf0000002000300000",
			["first", "second", "third"],
			id="a copy past the section",
		),
		pytest.param(
			"08b503480349082200f07af908bd00bf0000002000300000",
			_EVERY_OBJECT,
			id="a copy through a forwarding function",
		),
		pytest.param(
			"08b503480349082200f09af908bd00bf0000002000300000",
			_ALL_BUT_FIRST,
			id="a call to a function that does more than forward",
		),
		pytest.param(
			"08b504480449091a0a46044900f0f8f808bd00bf000000200800002000300000",
			_EVERY_OBJECT,
			id="a copy of a computed length",
		),
		pytest.param(_UNKNOWN_LENGTH_COPY, _ALL_BUT_FIRST, id="a copy of an unknown length"),
		pytest.param(
			"08b503480021102200f01af908bd00bf10000020",
			_EVERY_OBJECT,
			id="a zero fill of a section the image holds no bytes for",
		),
		pytest.param(
			"08b503480121102200f01af908bd00bf10000020",
			["first", "second", "third"],
			id="a fill with a value other than zero",
		),
		pytest.param(
			"08b503480021082200f01af908bd00bf00000020",
			["third", "zeroed"],
			id="a zero fill of a section the image holds bytes for",
		),
	],
)
def test_an_initialization_call_keeps_only_the_objects_it_leaves_at_their_image_value(
	code: str, unescaped: list[str]
) -> None:
	assert _unescaped(_program(code)) == unescaped


@pytest.mark.parametrize(
	("code", "objects", "unescaped"),
	[
		pytest.param(_ARGUMENT, _AFTER_FIRST, ["zeroed"], id="a label that starts no object"),
		pytest.param(
			_ARGUMENT, _OBJECTS, _ALL_BUT_FIRST, id="a label at an object among several types"
		),
		pytest.param(_ARGUMENT, _ARRAY, ["zeroed"], id="a label at an object of an array"),
		pytest.param(
			_WALK, _OBJECTS, _ALL_BUT_FIRST, id="an unknown offset from objects of several types"
		),
		pytest.param(_WALK, _ARRAY, ["zeroed"], id="an unknown offset from an array"),
		pytest.param(_UNKNOWN_LENGTH_COPY, _OBJECTS, ["zeroed"], id="a fill from a label"),
	],
)
def test_a_linker_label_lets_its_whole_section_escape_where_code_may_walk_it(
	code: str, objects: tuple[tuple[str, int, int, str | None], ...], unescaped: list[str]
) -> None:
	assert (
		_unescaped(_program(code, objects=objects, labels={"region_start": 0x2000_0000}))
		== unescaped
	)


@pytest.mark.parametrize(
	("code", "data_in_code", "sections", "unescaped"),
	[
		pytest.param(
			_BASE_ONLY, ((0x1008, 0x100C),), {}, _EVERY_OBJECT, id="a literal a load reads"
		),
		pytest.param(
			"704700bf00000020", ((0x1004, 0x1008),), {}, _ALL_BUT_FIRST, id="data in code"
		),
		pytest.param(
			_BASE_ONLY,
			(),
			{0x4000: (0x2000_0004).to_bytes(4, "little")},
			["first", "third", "zeroed"],
			id="a data section",
		),
		pytest.param(
			_BASE_ONLY,
			(),
			{0x4000: b"\x00" + (0x2000_0004).to_bytes(4, "little")},
			["first", "third", "zeroed"],
			id="a packed data section",
		),
	],
)
def test_a_word_of_data_lets_the_object_it_points_into_escape(
	code: str,
	data_in_code: tuple[tuple[int, int], ...],
	sections: Mapping[int, bytes],
	unescaped: list[str],
) -> None:
	assert _unescaped(_program(code, data_in_code=data_in_code, sections=sections)) == unescaped


@pytest.mark.parametrize(
	("code", "unescaped"),
	[
		pytest.param(
			"04009fe5021090e61eff2fe100000020",
			_ALL_BUT_FIRST,
			id="by a register the scan can't follow",
		),
		pytest.param(
			"08009fe50420a0e3021090e61eff2fe100000020",
			["first", "third", "zeroed"],
			id="by a constant register",
		),
	],
)
def test_an_arm_post_indexed_load_moves_its_base(code: str, unescaped: list[str]) -> None:
	assert _unescaped(_program(code, arm_code=((0x1000, 0x1000 + len(code) // 2),))) == unescaped


def test_a_branch_back_into_a_merge_point_joins_rather_than_widens() -> None:
	"""``cmp; beq else; merge: mov r0, r3; bl other; ...; else: adds r3, #4; b merge``."""
	assert _unescaped(_program("08b5044b002903d0184600f079f808bd0433f9e700000020")) == [
		"third",
		"zeroed",
	]


@pytest.mark.parametrize(
	"code",
	[
		pytest.param("08b502a000f07cf808bd00bf00000000", id="adr"),
		pytest.param("08b50ff2080000f07bf808bd0000000000000000", id="addw from pc"),
	],
)
def test_a_pc_relative_address_escapes_the_object_it_points_into(code: str) -> None:
	assert _unescaped(
		_program(
			code,
			objects=(*_OBJECTS, ("near", 0x100C, 4, "int"), ("far", 0x1010, 4, "int")),
			writable_code=True,
		)
	) == ["far", *_EVERY_OBJECT]


def _predicated_loads(count: int) -> str:
	"""``it eq; ldreq r0, =address`` per address, then ``b 1f; 1: movs r0, #0; bx lr``."""
	return (
		f"08bf{(0x4800 | (count + 1)).to_bytes(2, 'little').hex()}" * count
		+ "ffe700207047"
		+ "00bf"
		+ "".join((0x2000_0000 + 4 * index).to_bytes(4, "little").hex() for index in range(count))
	)


@pytest.mark.parametrize(
	("count", "unescaped"),
	[pytest.param(64, _EVERY_OBJECT, id="64"), pytest.param(65, [], id="65")],
)
def test_a_register_that_may_hold_too_many_addresses_lets_them_escape(
	count: int, unescaped: list[str]
) -> None:
	assert _unescaped(_program(_predicated_loads(count))) == unescaped


def test_a_loop_widens_an_address_to_the_objects_it_reached() -> None:
	assert _unescaped(
		_program(
			"08b5034804308842fcd100f079f808bd00000020",
			objects=(("table", 0x2000_0000, 0x40, "int [16]"), ("after", 0x2000_0100, 4, "int")),
		)
	) == ["after"]


@pytest.mark.parametrize(
	("code", "target"),
	[
		pytest.param(_BASE_ONLY, (0x2000_0000, 0x2000_0004), id="its base"),
		pytest.param("014b43f82100704700000020", (0x2000_0000, 0x2000_0004), id="an index"),
		pytest.param("014b5860704700bf00000020", (0x2000_0004, 0x2000_0008), id="a displacement"),
		pytest.param("014b83e80600704700000020", (0x2000_0000, 0x2000_0004), id="a store multiple"),
		pytest.param(
			"014b03e90600704708000020",
			(0x2000_0000, 0x2000_0004),
			id="a store multiple below its base",
		),
	],
)
def test_a_store_targets_the_object_its_address_was_derived_from(
	code: str, target: tuple[int, int]
) -> None:
	assert _escapes(_program(code)).store_targets == {Address(0x1002): frozenset({target})}


def test_overlapping_objects_escape_together() -> None:
	assert _escapes(
		_program(
			_BASE_ONLY,
			objects=(("outer", 0x2000_0000, 8, "long long int"), ("inner", 0x2000_0004, 4, "int")),
		)
	).unescaped == ((0x2000_0000, 0x2000_0008),)


def test_x86_images_have_no_unescaped_objects() -> None:
	assert escapes(
		build_program(
			Machine.EM_X86_64, objects=(("hook", 0x4000, bytes(8)),), writable=frozenset({0x4000})
		),
		(),
	) == Escapes(unescaped=(), store_targets={})
