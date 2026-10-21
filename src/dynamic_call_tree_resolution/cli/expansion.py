# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Expanding a build's indirect calls from its image."""

from typing import TYPE_CHECKING, assert_never

from dynamic_call_tree_resolution.call_sites import (
	narrowed_targets,
	per_caller_candidates,
	resolve,
)
from dynamic_call_tree_resolution.callgraph import (
	CallEdge,
	EdgeKind,
)
from dynamic_call_tree_resolution.cli.artifacts import (
	Artifacts,
	AssumedUnwritten,
	Expansion,
	Image,
	build_artifacts,
	loaded,
)
from dynamic_call_tree_resolution.cli.measures import (
	Targets,
	build_kconfig,
	code_measures,
	exception_priority_levels,
	image_function_names,
	own_thread_graphs,
	register_callers,
	thread_graph,
	unrecorded_sites,
)
from dynamic_call_tree_resolution.descriptors import load_descriptors
from dynamic_call_tree_resolution.field_narrowing import field_narrowings
from dynamic_call_tree_resolution.identity import (
	stack_name,
)
from dynamic_call_tree_resolution.loader import (
	line_spans,
)
from dynamic_call_tree_resolution.model import Address, Dead, Unreached
from dynamic_call_tree_resolution.points_to import null_slots, unresolved_slots
from dynamic_call_tree_resolution.report import (
	slot_counts,
)
from dynamic_call_tree_resolution.rtos import RtosChoice, rtos_model
from dynamic_call_tree_resolution.span import Span
from dynamic_call_tree_resolution.stack_analysis import (
	INDIRECT_CALLEE,
	expand_indirect_calls,
	frame_key,
)
from dynamic_call_tree_resolution.vsa.abi import normalized
from dynamic_call_tree_resolution.vsa.frames import (
	CodeMeasure,
	code_depths,
	code_measure,
)
from dynamic_call_tree_resolution.vsa.lattice import Known, Top
from dynamic_call_tree_resolution.vsa.links import linked_calls
from dynamic_call_tree_resolution.vsa.vectors import (
	exception_handlers,
	hardware_handlers,
)

if TYPE_CHECKING:
	from pathlib import Path

	from dynamic_call_tree_resolution.call_sites import ProgramResolution
	from dynamic_call_tree_resolution.field_narrowing import NarrowedSpan
	from dynamic_call_tree_resolution.model import CallSite, Program, RtosModel
	from dynamic_call_tree_resolution.stack_usage import StackUsage


def field_narrowed_spans(elf: Path, program: Program) -> tuple[NarrowedSpan, ...]:
	descriptors = load_descriptors(elf.parent / "descriptors.txt")
	return field_narrowings(
		program,
		descriptors,
		line_spans(elf, frozenset(site.location for site in descriptors.sites)),
	)


def elf_expansion(
	build_directory: Path,
	elf: Path,
	*,
	narrow_by_signature: bool,
	narrow_by_field: bool,
	rtos: RtosChoice,
	assume_unwritten: tuple[str, ...] = (),
) -> Expansion:
	"""Expand a build directory's indirect calls against its ELF image."""
	image = loaded(elf)
	return expand_from_elf(
		build_artifacts(build_directory, image),
		image,
		narrow_by_signature=narrow_by_signature,
		narrow_by_field=narrow_by_field,
		rtos=rtos,
		assume_unwritten=assume_unwritten,
	)


def expand_from_elf(
	artifacts: Artifacts,
	image: Image,
	*,
	narrow_by_signature: bool,
	narrow_by_field: bool,
	rtos: RtosChoice,
	assume_unwritten: tuple[str, ...] = (),
) -> Expansion:
	image_functions = image_function_names(artifacts, image)
	in_image_edges = tuple(
		edge for edge in artifacts.edges if frame_key(edge.caller) in image_functions
	)
	indirect_sites = sum(edge.callee == INDIRECT_CALLEE for edge in in_image_edges)
	model = rtos_model(image.program, rtos, build_kconfig(image.elf))
	spans = _unwritten_spans(image.program, assume_unwritten)
	resolution = resolve(image.program, model, assumed_unwritten=spans)
	narrowed_by_field = field_narrowed_spans(image.elf, image.program) if narrow_by_field else ()
	hardware_roots = frozenset(
		stack_name(image.program, image.names, handler)
		for handler in hardware_handlers(image.program)
	)
	exceptions = (
		{
			number: frame_key(stack_name(image.program, image.names, handler))
			for number, handler in exception_handlers(image.program).items()
		}
		if model.interrupt_stack is not None
		else {}
	)
	targets_by_caller, fallback = per_caller_candidates(
		image.program,
		resolution.sites,
		resolution.assignments,
		narrow_by_signature=narrow_by_signature,
		narrowed_by_field=narrowed_by_field,
		names=image.names,
		fallback=resolution.fallback,
	)
	narrowed_by_caller = narrowed_targets(
		image.program,
		resolution.sites,
		resolution.assignments,
		narrow_by_signature=narrow_by_signature,
		narrowed_by_field=narrowed_by_field,
		names=image.names,
		fallback=resolution.fallback,
	)
	unresolved = unresolved_slots(image.program, resolution.assignments)
	measure = code_measure(image.program)
	register = register_callers(image, measure, resolution.sites, artifacts.frames)
	unrecorded = unrecorded_sites(image, resolution.sites, (*in_image_edges, *register.edges))
	expanded = expand_indirect_calls(
		(*artifacts.edges, *register.edges, *unrecorded),
		targets_by_caller,
		fallback,
		narrowed_by_caller=narrowed_by_caller,
	)
	binary = _binary_edges(image, measure, expanded, artifacts.frames)
	threads_edges, threads_frames = thread_graph(
		image,
		model,
		(*in_image_edges, *binary, *register.edges),
		(*artifacts.frames, *register.frames),
	)
	code = code_measures(
		image,
		measure,
		(*expanded, *binary, *threads_edges),
		(*artifacts.frames, *threads_frames, *register.frames),
		hardware_roots | frozenset(exceptions.values()),
		frozenset(),
	)
	return Expansion(
		expanded=(*expanded, *binary, *code.edges, *threads_edges),
		in_image_edges=(*in_image_edges, *unrecorded, *threads_edges),
		frames=(*artifacts.frames, *threads_frames, *register.frames, *code.frames),
		threads=frozenset(thread.name for thread in model.threads)
		| frozenset(thread.name for thread in model.system_threads),
		hardware_roots=hardware_roots,
		exceptions=exceptions,
		priority_levels=exception_priority_levels(image.elf, exceptions),
		own_edges={
			thread: (*expanded, *binary, *code.edges, *own)
			for thread, own in own_thread_graphs(
				image,
				model,
				resolution,
				(*in_image_edges, *unrecorded, *binary),
				Targets(
					by_caller=targets_by_caller,
					narrowed_by_caller=narrowed_by_caller,
					fallback=fallback,
					narrowed_by_field=narrowed_by_field,
					narrow_by_signature=narrow_by_signature,
				),
			).items()
		},
		indirect_sites=indirect_sites,
		program=image.program,
		rtos=model,
		resolution=resolution,
		counts=slot_counts(
			resolution.assignments, unresolved, null_slots(image.program, resolution.assignments)
		),
		unresolved=unresolved,
		image_functions=image_functions,
		membership=artifacts.membership,
		dropped=artifacts.dropped,
		phantom_libcalls=artifacts.phantom_libcalls,
		assumed_unwritten=_assumed_unwritten(image, model, resolution, assume_unwritten),
	)


def _unwritten_spans(program: Program, names: tuple[str, ...]) -> tuple[Span, ...]:
	return tuple(span for name in names for span in (_unwritten_span(program, name),))


def _unwritten_span(program: Program, name: str) -> Span:
	match tuple(obj for obj in program.objects.values() if obj.name == name):
		case (found,):
			return Span(start=found.address, end=Address(found.address + found.size))
		case ():
			raise ValueError(f"--assume-unwritten: no data object is named {name}")
		case several:
			raise ValueError(
				f"--assume-unwritten: {len(several)} data objects are named {name}, at "
				f"{', '.join(hex(obj.address) for obj in several)}"
			)


def _assumed_unwritten(
	image: Image, model: RtosModel, resolution: ProgramResolution, names: tuple[str, ...]
) -> AssumedUnwritten:
	if not names:
		return AssumedUnwritten(objects=(), contradicted=(), callers=frozenset(), thread_callers={})
	baseline = resolve(image.program, model)
	return AssumedUnwritten(
		objects=names,
		contradicted=tuple(
			name
			for name in names
			for span in (_unwritten_span(image.program, name),)
			if any(
				span.start - image.program.pointer_size < key < span.end
				for key in resolution.written
			)
		),
		callers=_changed_callers(image, baseline.sites, resolution.sites),
		thread_callers={
			thread: _changed_callers(image, baseline.threads[thread].sites, own.sites)
			for thread, own in resolution.threads.items()
		},
	)


def _changed_callers(
	image: Image, before: tuple[CallSite, ...], after: tuple[CallSite, ...]
) -> frozenset[str]:
	targets = {(site.caller_address, site.site_address): _target_key(site) for site in before}
	return frozenset(
		frame_key(stack_name(image.program, image.names, site.caller_address))
		for site in after
		if targets.get((site.caller_address, site.site_address)) != _target_key(site)
	)


def _target_key(site: CallSite) -> tuple[str, frozenset[Address]]:
	match site.target:
		case Known(values=values):
			return "known", values
		case Top() | Unreached() | Dead() as other:
			return type(other).__name__, frozenset()
		case _ as unreachable:
			assert_never(unreachable)


def _call_pair(edge: CallEdge) -> tuple[str, str]:
	return edge.caller, edge.callee


def _binary_edges(
	image: Image, measure: CodeMeasure, edges: tuple[CallEdge, ...], frames: tuple[StackUsage, ...]
) -> tuple[CallEdge, ...]:
	framed = frozenset(frame_key(frame.function) for frame in frames)
	known = frozenset((frame_key(edge.caller), frame_key(edge.callee)) for edge in edges)
	calls = tuple(
		(caller, callee)
		for call in linked_calls(image.program)
		if call.caller is not None
		for caller in (normalized(call.caller, image.program.machine),)
		for callee in (normalized(call.callee, image.program.machine),)
		if caller in image.names and callee in image.names
	)
	measured = code_depths(
		measure,
		frozenset(caller for caller, _ in calls if frame_key(image.names[caller]) not in framed),
	).keys()
	return tuple(
		sorted(
			{
				CallEdge(
					caller=image.names[caller], callee=image.names[callee], kind=EdgeKind.BINARY
				)
				for caller, callee in calls
				if caller not in measured
				and (frame_key(image.names[caller]), frame_key(image.names[callee])) not in known
			},
			key=_call_pair,
		)
	)
