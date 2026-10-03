# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The descriptors plugin's records, joined to the ``.ci`` indirect calls (#158)."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import pytest

from dynamic_call_tree_resolution.callgraph import IndirectCall
from dynamic_call_tree_resolution.descriptors import (
	DescribedCall,
	Descriptors,
	Field,
	Function,
	Null,
	Parameter,
	Site,
	Store,
	Variable,
	described_calls,
	load_descriptors,
)
from dynamic_call_tree_resolution.model import SourceLocation
from dynamic_call_tree_resolution.stack_analysis import frame_key

if TYPE_CHECKING:
	from pathlib import Path


def test_a_call_takes_the_descriptor_its_own_units_plugin_sites_agree_on(tmp_path: Path) -> None:
	application = tmp_path / "CMakeFiles" / "app.dir" / "src"
	library = tmp_path / "zephyr" / "CMakeFiles" / "library.dir" / "src"
	application.mkdir(parents=True)
	library.mkdir(parents=True)
	(application / "a.c.ci").write_text(
		'graph: { title: "a.c"\n'
		'edge: { sourcename: "f.constprop.1" targetname: "__indirect_call" label: "/app/a.c:3:9" }\n'
		'edge: { sourcename: "/app/a.c:g" targetname: "__indirect_call" label: "/app/a.h:7:2" }\n'
		'edge: { sourcename: "/app/a.c:g" targetname: "h" }\n'
		"}\n"
	)
	(library / "a.c.ci").write_text(
		'graph: { title: "a.c"\n'
		'edge: { sourcename: "f" targetname: "__indirect_call" label: "/library/a.c:3:9" }\n'
		"}\n"
	)
	assert described_calls(
		Descriptors(
			sites=(
				Site(
					unit="CMakeFiles/app.dir/src/a.c",
					function="f.constprop.0",
					location=SourceLocation(file="a.c", line=3, column=9),
					callee=Field(record="struct ops", member="run"),
				),
				Site(
					unit="CMakeFiles/app.dir/src/a.c",
					function="g",
					location=SourceLocation(file="a.h", line=7, column=2),
					callee=Parameter(name="cb"),
				),
				Site(
					unit="CMakeFiles/app.dir/src/a.c",
					function="g",
					location=SourceLocation(file="a.h", line=7, column=2),
					callee=Variable(symbol="hook"),
				),
				Site(
					unit="zephyr/CMakeFiles/library.dir/src/a.c",
					function="g",
					location=SourceLocation(file="a.h", line=9, column=2),
					callee=Null(),
				),
			),
			stores=(),
		),
		tmp_path,
	) == (
		DescribedCall(
			unit="CMakeFiles/app.dir/src/a.c",
			call=IndirectCall(
				caller="f.constprop.1", location=SourceLocation(file="a.c", line=3, column=9)
			),
			callee=Field(record="struct ops", member="run"),
		),
		DescribedCall(
			unit="CMakeFiles/app.dir/src/a.c",
			call=IndirectCall(
				caller="/app/a.c:g", location=SourceLocation(file="a.h", line=7, column=2)
			),
			callee=None,
		),
		DescribedCall(
			unit="zephyr/CMakeFiles/library.dir/src/a.c",
			call=IndirectCall(caller="f", location=SourceLocation(file="a.c", line=3, column=9)),
			callee=None,
		),
	)


@pytest.fixture(scope="module")
def sensor_two_impl(zephyr_fixtures: Path) -> tuple[Descriptors, tuple[DescribedCall, ...]]:
	build_directory = zephyr_fixtures / "sensor-two-impl"
	descriptors = load_descriptors(build_directory / "zephyr" / "descriptors.txt")
	return descriptors, described_calls(descriptors, build_directory)


@pytest.mark.image
def test_every_indirect_call_in_the_ci_joins_exactly_one_plugin_site(
	sensor_two_impl: tuple[Descriptors, tuple[DescribedCall, ...]],
) -> None:
	descriptors, calls = sensor_two_impl
	assert (
		len(descriptors.sites),
		Counter((site.unit, frame_key(site.function), site.location) for site in descriptors.sites)
		== Counter((call.unit, frame_key(call.call.caller), call.call.location) for call in calls),
		Counter(type(call.callee) for call in calls),
		[
			unit
			for unit in (
				*(site.unit for site in descriptors.sites),
				*(store.unit for store in descriptors.stores),
			)
			if "source/" in unit
		],
	) == (60, True, Counter({Field: 30, Parameter: 27, Variable: 3}), [])


@pytest.mark.image
def test_the_device_api_calls_load_their_callee_from_a_field(
	sensor_two_impl: tuple[Descriptors, tuple[DescribedCall, ...]],
) -> None:
	_, calls = sensor_two_impl
	assert sorted(
		(frame_key(call.call.caller), call.call.location.file, call.call.location.line, call.callee)
		for call in calls
		if (call.call.location.file, call.call.location.line)
		in {("i2c.h", 1020), ("bmi160.c", 1048), ("device.c", 23)}
	) == [
		("bmi160_init", "bmi160.c", 1048, Field(record="struct bmi160_bus_io", member="ready")),
		(
			"bmi160_read_i2c",
			"i2c.h",
			1020,
			Field(record="struct i2c_driver_api", member="transfer"),
		),
		(
			"bmi160_write_i2c",
			"i2c.h",
			1020,
			Field(record="struct i2c_driver_api", member="transfer"),
		),
		("do_device_init", "device.c", 23, Field(record="struct device_ops", member="init")),
		(
			"i2c_emul_transfer",
			"i2c.h",
			1020,
			Field(record="struct i2c_driver_api", member="transfer"),
		),
		("i2c_write", "i2c.h", 1020, Field(record="struct i2c_driver_api", member="transfer")),
		(
			"i2c_write_read",
			"i2c.h",
			1020,
			Field(record="struct i2c_driver_api", member="transfer"),
		),
	]


@pytest.mark.image
def test_a_store_names_what_it_stores_into_and_what_it_stores(
	sensor_two_impl: tuple[Descriptors, tuple[DescribedCall, ...]],
) -> None:
	descriptors, _ = sensor_two_impl
	assert (
		len(descriptors.stores),
		{
			Store(
				unit="zephyr/kernel/CMakeFiles/kernel.dir/work.c",
				function="k_work_init",
				location=SourceLocation(file="work.c", line=159, column=8),
				place=Field(record="struct k_work", member="handler"),
				value=Parameter(name="handler"),
			),
			Store(
				unit="zephyr/CMakeFiles/zephyr.dir/lib/os/printk.c",
				function="vprintk_core",
				location=SourceLocation(file="printk.c", line=123, column=7),
				place=Field(record="struct __file", member="put"),
				value=Function(symbol="char_out"),
			),
			Store(
				unit="zephyr/CMakeFiles/zephyr.dir/lib/os/printk.c",
				function="__printk_hook_install",
				location=SourceLocation(file="printk.c", line=59, column=12),
				place=Variable(symbol="_char_out"),
				value=Parameter(name="fn"),
			),
		}
		<= set(descriptors.stores),
	) == (11, True)
