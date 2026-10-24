# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Which writable objects only the code that names them can reach (C's pointer provenance)."""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Final, assert_never

from salix import Struct

from dynamic_call_tree_resolution.model import Address, InstructionFamily
from dynamic_call_tree_resolution.vsa.cfg import predecessors
from dynamic_call_tree_resolution.vsa.escape.image import (
	Image,
	containing,
	data_escapes,
	escape_image,
	reached_objects,
	sections_objects,
)
from dynamic_call_tree_resolution.vsa.escape.transfer import (
	RETURNED,
	Effects,
	Scope,
	step,
)
from dynamic_call_tree_resolution.vsa.escape.values import (
	UNKNOWN,
	Registers,
	join,
	size,
)

if TYPE_CHECKING:
	from collections.abc import Callable, Iterator, Mapping
	from concurrent.futures import Executor

	from dynamic_call_tree_resolution.model import Function, Program
	from dynamic_call_tree_resolution.span import Span
	from dynamic_call_tree_resolution.vsa.cfg import Block, FunctionGraph


class Escapes(Struct):
	"""What the code lets escape, which bounds what a store the analysis can't place may write."""

	unescaped: tuple[Span, ...]
	"""Writable objects whose address no escaped value can reach, sorted."""
	store_targets: Mapping[Address, frozenset[Span]]
	"""The unescaped objects each store, by address, may write through an address the code derived
	from them."""


NO_ESCAPES: Final = Escapes(unescaped=(), store_targets={})


class _Scan(Struct):
	escaped: frozenset[Span]
	stores: Mapping[Address, frozenset[Span]]
	literals: frozenset[int]


def escapes(
	program: Program,
	graphs: tuple[FunctionGraph, ...],
	executor: Executor | None = None,
) -> Escapes:
	match program.machine.family:
		case InstructionFamily.X86:
			return NO_ESCAPES
		case InstructionFamily.ARM:
			return _arm_escapes(escape_image(program, graphs), graphs, executor)
		case _ as unreachable:
			assert_never(unreachable)


def _arm_escapes(
	image: Image,
	graphs: tuple[FunctionGraph, ...],
	executor: Executor | None,
) -> Escapes:
	scans = (
		tuple(map(partial(_scan_graph, image), graphs))
		if executor is None
		else tuple(executor.map(partial(_scan_graph, image), graphs))
	)
	escaped = frozenset(chain.from_iterable(scan.escaped for scan in scans)) | data_escapes(
		image, frozenset(chain.from_iterable(scan.literals for scan in scans))
	)
	return Escapes(
		unescaped=tuple(span for span in image.spans if span not in escaped),
		store_targets={
			site: targets
			for scan in scans
			for site, reached in scan.stores.items()
			if (targets := reached - escaped)
		},
	)


def _scan_graph(image: Image, graph: FunctionGraph) -> _Scan:
	return _scan(image, graph.function, graph.blocks)


def _scan(image: Image, function: Function, blocks: tuple[Block, ...]) -> _Scan:
	by_start = {block.start: block for block in blocks}
	scope = Scope(
		blocks=frozenset(by_start),
		returned=frozenset()
		if function.signature is not None and function.signature.return_type == "void"
		else RETURNED,
	)
	carriers = _carriers(blocks, by_start)
	in_states: dict[Address, Registers] = {}
	effects: list[Effects] = []
	for seed in by_start:
		if seed in in_states:
			continue
		in_states[seed] = UNKNOWN
		worklist = [seed]
		while worklist:
			block = by_start[worklist.pop()]
			registers = in_states[block.start]
			for instruction in block.instructions:
				registers, effect = step(image, scope, instruction, registers, block.successors)
				effects.append(effect)
			for successor in block.successors:
				joined, overflow = join(
					in_states.get(successor),
					registers,
					carries=carriers.get(_Edge(source=block.start, destination=successor)),
				)
				if size(overflow):
					effects.append(Effects(escaped=overflow))
				if joined != in_states.get(successor):
					in_states[successor] = joined
					worklist.append(successor)
	return _scanned(image, tuple(effects))


class _Edge(Struct):
	"""An edge of a function's control-flow graph."""

	source: Address
	destination: Address


def _carriers(
	blocks: tuple[Block, ...], by_start: Mapping[Address, Block]
) -> Mapping[_Edge, Callable[[int], bool]]:
	"""For each edge back into a loop, which materializations the loop carries.

	Those outside the loop body, when the loop's head dominates the edge; none of a loop that
	control can enter elsewhere.
	"""
	sites = {
		instruction.address: block.start for block in blocks for instruction in block.instructions
	}
	inbound = predecessors(by_start)
	reachable = _reached(by_start, blocks[0].start, None)
	return {
		edge: partial(_outside, sites, _loop_body(inbound, edge.source, edge.destination))
		if edge.source in reachable
		and edge.source not in _reached(by_start, blocks[0].start, edge.destination)
		else _carries_nothing
		for edge in _back_edges(by_start)
	}


def _outside(sites: Mapping[int, Address], body: frozenset[Address], site: int) -> bool:
	return sites[site] not in body


def _carries_nothing(_: int) -> bool:
	return False


def _reached(
	by_start: Mapping[Address, Block], root: Address, avoided: Address | None
) -> frozenset[Address]:
	"""The blocks control reaches from the root without passing through the avoided one."""
	seen: set[Address] = set()
	stack = [root]
	while stack:
		start = stack.pop()
		if start not in seen and start != avoided:
			seen.add(start)
			stack.extend(by_start[start].successors)
	return frozenset(seen)


def _loop_body(
	inbound: Mapping[Address, tuple[Address, ...]], source: Address, head: Address
) -> frozenset[Address]:
	"""The blocks that reach the edge's source without passing through the loop head."""
	seen: set[Address] = {head}
	stack = [source]
	while stack:
		start = stack.pop()
		if start not in seen:
			seen.add(start)
			stack.extend(inbound.get(start, ()))
	return frozenset(seen)


class _Visit(Struct):
	"""A block on the depth-first path, with the successors it has yet to follow."""

	start: Address
	successors: Iterator[Address]


def _back_edges(by_start: Mapping[Address, Block]) -> frozenset[_Edge]:
	"""The edges into a block on the depth-first path to their source, which every loop has."""
	finished: set[Address] = set()
	on_path: set[Address] = set()
	edges: set[_Edge] = set()
	for root in by_start:
		if root in finished:
			continue
		on_path.add(root)
		stack = [_Visit(root, iter(by_start[root].successors))]
		while stack:
			visit = stack[-1]
			successor = next(visit.successors, None)
			if successor is None:
				stack.pop()
				on_path.discard(visit.start)
				finished.add(visit.start)
			elif successor in on_path:
				edges.add(_Edge(source=visit.start, destination=successor))
			elif successor not in finished:
				on_path.add(successor)
				stack.append(_Visit(successor, iter(by_start[successor].successors)))
	return frozenset(edges)


def _scanned(image: Image, effects: tuple[Effects, ...]) -> _Scan:
	stores: defaultdict[Address, set[Span]] = defaultdict(set)
	for effect in effects:
		if effect.site is not None:
			stores[effect.site].update(reached_objects(image, effect.store))
	return _Scan(
		escaped=frozenset(
			chain(
				chain.from_iterable(reached_objects(image, effect.escaped) for effect in effects),
				chain.from_iterable(
					containing(image, derived.address)
					| sections_objects(image, image.bounded, derived.address)
					| sections_objects(image, image.bounded, derived.origin)
					for effect in effects
					for derived in chain(effect.filled.exact, effect.filled.indexed)
				),
				chain.from_iterable(effect.overwritten for effect in effects),
			)
		),
		stores={site: frozenset(targets) for site, targets in stores.items() if targets},
		literals=frozenset(effect.literal for effect in effects if effect.literal is not None),
	)
