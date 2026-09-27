# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Tests for :mod:`dynamic_call_tree_resolution.callgraph`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dynamic_call_tree_resolution import CallEdge
from dynamic_call_tree_resolution.callgraph import load_callgraph, parse_callgraph

if TYPE_CHECKING:
	from pathlib import Path

VCG_CONTENT = """\
graph: {
title: "callgraph"
node: { title: "main" label: "main\\n/path/to/main.c:13:1\\n8 bytes (static)\\n0 dynamic objects" }
node: { title: "can_send" label: "can_send\\n" shape : ellipse }
edge: { sourcename: "main" targetname: "can_send" label: "/path/to/main.c:24\\n" }
edge: { sourcename: "can_send" targetname: "z_impl_can_send" label: "16 bytes (static)\\n" }
}
"""


def test_parse_callgraph(tmp_path: Path) -> None:
	callgraph_file = tmp_path / "canbus.c.ci"
	callgraph_file.write_text(VCG_CONTENT)
	assert parse_callgraph(callgraph_file) == (
		CallEdge(caller="main", callee="can_send"),
		CallEdge(caller="can_send", callee="z_impl_can_send"),
	)


def test_parse_callgraph_without_edges(tmp_path: Path) -> None:
	callgraph_file = tmp_path / "empty.ci"
	callgraph_file.write_text('graph: { title: "callgraph" node: { title: "leaf" } }\n')
	assert parse_callgraph(callgraph_file) == ()


def test_parse_callgraph_rejects_malformed_graph(tmp_path: Path) -> None:
	callgraph_file = tmp_path / "bad.ci"
	callgraph_file.write_text("graph: { edge: { sourcename main }\n")
	with pytest.raises(ValueError, match="expected ':' after 'sourcename'"):
		parse_callgraph(callgraph_file)


def test_parse_callgraph_rejects_graph_without_braces(tmp_path: Path) -> None:
	callgraph_file = tmp_path / "bad.ci"
	callgraph_file.write_text('graph: edge: { sourcename: "main" }\n')
	with pytest.raises(ValueError, match=r"expected 'graph: {'"):
		parse_callgraph(callgraph_file)


def test_parse_callgraph_rejects_edge_without_target(tmp_path: Path) -> None:
	callgraph_file = tmp_path / "bad.ci"
	callgraph_file.write_text('graph: { edge: { sourcename: "main" } }\n')
	with pytest.raises(ValueError, match="targetname"):
		parse_callgraph(callgraph_file)


def test_parse_callgraph_rejects_entry_without_colon(tmp_path: Path) -> None:
	callgraph_file = tmp_path / "bad.ci"
	callgraph_file.write_text('graph: { edge sourcename: "main" }\n')
	with pytest.raises(ValueError, match="expected ':' after 'edge'"):
		parse_callgraph(callgraph_file)


def test_load_callgraph_collects_all_ci_files(tmp_path: Path) -> None:
	build_directory = tmp_path / "build"
	nested = build_directory / "CMakeFiles" / "app.dir"
	nested.mkdir(parents=True)
	(nested / "a.c.ci").write_text('graph: { edge: { sourcename: "a" targetname: "b" } }\n')
	(nested / "b.c.ci").write_text('graph: { edge: { sourcename: "b" targetname: "c" } }\n')
	assert load_callgraph(build_directory) == (
		CallEdge(caller="a", callee="b"),
		CallEdge(caller="b", callee="c"),
	)
