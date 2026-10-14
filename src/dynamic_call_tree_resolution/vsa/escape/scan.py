# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Which writable objects only the code that names them can reach (C's pointer provenance)."""

from __future__ import annotations

from collections import defaultdict
from itertools import chain
from typing import TYPE_CHECKING, assert_never

from salix import Struct

from dynamic_call_tree_resolution.model import Address, InstructionFamily
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
	from collections.abc import Mapping

	from dynamic_call_tree_resolution.model import Function, Program
	from dynamic_call_tree_resolution.vsa.cfg import Block
	from dynamic_call_tree_resolution.vsa.escape.image import Span


class Escapes(Struct):
	"""What the code lets escape, which bounds what a store the analysis can't place may write."""

	unescaped: tuple[Span, ...]
	"""Writable objects whose address no escaped value can reach, sorted."""
	store_targets: Mapping[Address, frozenset[Span]]
	"""The unescaped objects each store, by address, may write through an address the code derived
	from them."""


class _Scan(Struct):
	escaped: frozenset[Span]
	stores: Mapping[Address, frozenset[Span]]
	literals: frozenset[int]


def escapes(program: Program, graphs: tuple[tuple[Function, tuple[Block, ...]], ...]) -> Escapes:
	match program.machine.family:
		case InstructionFamily.X86:
			return Escapes(unescaped=(), store_targets={})
		case InstructionFamily.ARM:
			return _arm_escapes(escape_image(program, graphs), graphs)
		case _ as unreachable:
			assert_never(unreachable)


def _arm_escapes(image: Image, graphs: tuple[tuple[Function, tuple[Block, ...]], ...]) -> Escapes:
	scans = tuple(_scan(image, function, blocks) for function, blocks in graphs)
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


def _scan(image: Image, function: Function, blocks: tuple[Block, ...]) -> _Scan:
	by_start = {block.start: block for block in blocks}
	scope = Scope(
		blocks=frozenset(by_start),
		returned=frozenset()
		if function.signature is not None and function.signature.return_type == "void"
		else RETURNED,
	)
	back = _back_edges(by_start)
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
					in_states.get(successor), registers, back=(block.start, successor) in back
				)
				if size(overflow):
					effects.append(Effects(escaped=overflow))
				if joined != in_states.get(successor):
					in_states[successor] = joined
					worklist.append(successor)
	return _scanned(image, tuple(effects))


def _back_edges(by_start: Mapping[Address, Block]) -> frozenset[tuple[Address, Address]]:
	"""The edges into a block on the depth-first path to their source, which every loop has."""
	finished: set[Address] = set()
	on_path: set[Address] = set()
	edges: set[tuple[Address, Address]] = set()
	for root in by_start:
		if root in finished:
			continue
		on_path.add(root)
		stack = [(root, iter(by_start[root].successors))]
		while stack:
			start, successors = stack[-1]
			successor = next(successors, None)
			if successor is None:
				stack.pop()
				on_path.discard(start)
				finished.add(start)
			elif successor in on_path:
				edges.add((start, successor))
			elif successor not in finished:
				on_path.add(successor)
				stack.append((successor, iter(by_start[successor].successors)))
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
