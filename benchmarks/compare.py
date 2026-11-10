# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Compare dctr between a base revision and the working tree: its outputs, its work, its CPU time.

Every run is a process of its own, with ``PYTHONPATH`` naming the tree it runs: the base
revision's ``src`` as ``git archive`` extracts it under ``.camas/benchmarks``, or the working
tree's. Both trees run on this interpreter, with the working tree's dependencies.
``benchmarks.measure`` imports the analysis's functions by name, so a base must still have them.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import shlex
import statistics
import sys
import tarfile
import tempfile
from collections import Counter
from functools import partial
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal

import msgspec
from cyclopts import App, Parameter, validators
from salix import Struct

from benchmarks.measurements import CpuTime, Work

if TYPE_CHECKING:
	from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping

_ROOT: Final = Path(__file__).resolve().parent.parent
_STORE: Final = _ROOT / ".camas" / "benchmarks"
_PEXPLORER_TESTDATA: Final = _ROOT / "references" / "pexplorer" / "testdata" / "elf_testdata"
_RTOS_TESTBEDS: Final = ("sensor-threads", "synchronization", "logger")
_SLOW_IMAGES: Final = frozenset(
	("prusa_buddy_boot_64", "zswatch_nrf5340_070", "zswatch_nrf5340_080", "zswatch_nrf5340_081")
)
_LOAD_POLL_SECONDS: Final = 15
_LOAD_POLLS: Final = 40
_DCTR: Final = "from dynamic_call_tree_resolution.cli import main; main()"
_IMPORTED_FROM: Final = "import dynamic_call_tree_resolution as dctr; print(dctr.__file__)"

_BASE: Final = Parameter(help="the revision to compare the working tree against")
_MEASURED_IMAGE: Final = Parameter(
	help="a testbed image to measure, by its fixture or testdata name; repeat it for each one. "
	"Without one, the RTOS testbeds: " + ", ".join(_RTOS_TESTBEDS)
)
_JOBS: Final = Parameter(
	help="how many processes to run at once", validator=validators.Number(gte=1)
)

app = App(name="compare", help=__doc__)


class _Tree(Struct):
	name: str
	source: Path
	revision: str | None
	"""The commit the tree was extracted from, whose outputs can be reused."""


class _Trees(Struct):
	base: _Tree
	head: _Tree


class _Image(Struct):
	name: str
	elf: Path
	build: Path | None
	"""The build directory whose .su and .ci artifacts ``dctr stack`` reads."""


class _DctrCommand(Struct):
	output: str
	"""The name the command's output is kept under, in each tree's directory of outputs."""
	elf: Path
	arguments: tuple[str, ...]


class _Completed(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
	exit_status: int
	stdout: bytes
	stderr: bytes


class _KeptOutput(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
	key: str
	"""What the output depends on besides the revision."""
	output: _Completed


class _OutputComparison(Struct):
	command: _DctrCommand
	identical: bool


class _Pair(Struct):
	base_seconds: float
	head_seconds: float
	load_before: float
	load_after: float


@app.command  # type: ignore[misc]
def outputs(
	base: Annotated[str, _BASE] = "main",
	*,
	image: Annotated[
		tuple[str, ...],
		Parameter(
			help="a testbed image to check, by its fixture or testdata name; repeat it for each "
			"one. Without one, every testbed image but the slow ones"
		),
	] = (),
	all_images: Annotated[
		bool,
		Parameter(
			name="--all",
			help="check the slow images too: " + ", ".join(sorted(_SLOW_IMAGES)) + ", which take "
			"ten minutes or more each where the rest take seconds",
		),
	] = False,
	jobs: Annotated[int, _JOBS] = 4,
) -> None:
	"""Exit 1 unless both trees print the same bytes and exit the same for each testbed image.

	The commands are ``dctr analyze --json`` for every image and ``dctr stack --json`` for every
	image with a build directory. The images are $DCTR_FIXTURES' builds and pexplorer's testdata,
	once ``camas pexplorer_testdata`` has fetched it. Each tree's outputs are kept under
	.camas/benchmarks/outputs, and the base revision's are reused until its ELF or the environment
	changes; clear them after rebuilding a build directory in place.

	Raises:
		SystemExit: when an output differs.
	"""
	chosen = _chosen(image, partial(_checked_by_default, all_images=all_images))
	comparisons = asyncio.run(_outputs(base, chosen, jobs))
	print("\n".join(map(_comparison_line, comparisons)))
	differing = sum(not comparison.identical for comparison in comparisons)
	if differing:
		raise SystemExit(f"{differing} of {len(comparisons)} outputs differ")


@app.command  # type: ignore[misc]
def work(
	base: Annotated[str, _BASE] = "main",
	*,
	image: Annotated[tuple[str, ...], _MEASURED_IMAGE] = (),
	jobs: Annotated[int, _JOBS] = 4,
) -> None:
	"""Print the instructions the interpreter transferred and the functions it analyzed.

	Machine load can't change either count, so the runs share the machine.
	"""
	print("\n".join(asyncio.run(_work(base, _chosen(image, _rtos_testbed), jobs))))


@app.command  # type: ignore[misc]
def cpu(
	base: Annotated[str, _BASE] = "main",
	*,
	image: Annotated[tuple[str, ...], _MEASURED_IMAGE] = (),
	pairs: Annotated[
		int,
		Parameter(help="how many runs of each tree per image", validator=validators.Number(gte=1)),
	] = 6,
	load_below: Annotated[
		float | None,
		Parameter(
			help="wait before each pair for the 1-minute load average to fall below this. Without "
			"it, the pairs run at once"
		),
	] = None,
	stage: Annotated[
		Literal["analysis", "load"],
		Parameter(help="what to time: the analysis, or loading the image before it"),
	] = "analysis",
) -> None:
	"""Print the process CPU seconds of the analysis or the load, in interleaved pairs of runs.

	The pairs alternate which tree runs first, so both trees share whatever load the machine
	carries. Every run's line carries the load average before and after its pair, and each image
	ends with every tree's minimum, median and maximum, and the change of the medians.
	"""
	asyncio.run(_cpu(base, _chosen(image, _rtos_testbed), pairs, load_below, stage))


async def _outputs(base: str, images: Iterable[_Image], jobs: int) -> tuple[_OutputComparison, ...]:
	trees = await _trees(base)
	limit = asyncio.Semaphore(jobs)
	commands = chain.from_iterable(map(_dctr_commands, images))
	return tuple(await asyncio.gather(*(_compared(trees, command, limit) for command in commands)))


async def _work(base: str, images: Iterable[_Image], jobs: int) -> list[str]:
	trees = await _trees(base)
	limit = asyncio.Semaphore(jobs)
	return await asyncio.gather(*(_work_line(trees, image, limit) for image in images))


async def _cpu(
	base: str,
	images: Iterable[_Image],
	pairs: int,
	load_below: float | None,
	stage: Literal["analysis", "load"],
) -> None:
	trees = await _trees(base)
	for image in images:
		measured = await _pairs(trees, image, pairs, load_below, stage)
		print(_cpu_summary(trees, image, measured), flush=True)


async def _trees(base: str) -> _Trees:
	revision = (await _git("rev-parse", "--verify", f"{base}^{{commit}}")).strip()
	source = _STORE / "trees" / revision / "src"
	if not source.is_dir():
		await _extract(revision, source)
	trees = _Trees(
		_Tree(f"{base}@{revision[:12]}", source, revision),
		_Tree("working-tree", _ROOT / "src", None),
	)
	print(f"base {trees.base.name}, head {trees.head.name}", flush=True)
	await _imports_from(trees.base)
	await _imports_from(trees.head)
	return trees


async def _extract(revision: str, source: Path) -> None:
	archive = await _output(("git", "archive", revision, "src"))
	source.parent.mkdir(parents=True, exist_ok=True)
	with (
		tempfile.TemporaryDirectory(dir=source.parent) as scratch,
		tarfile.open(fileobj=io.BytesIO(archive)) as tar,
	):
		tar.extractall(scratch, filter="data")
		(Path(scratch) / "src").rename(source)


async def _imports_from(tree: _Tree) -> None:
	imported = Path((await _python(tree, "-c", _IMPORTED_FROM)).decode().strip())
	if not imported.is_relative_to(tree.source.resolve()):
		raise SystemExit(f"{tree.name} imports dctr from {imported}, not from {tree.source}")


def _chosen(names: Collection[str], default: Callable[[_Image], bool]) -> tuple[_Image, ...]:
	images = tuple(_testbed_images())
	seen = Counter(image.name for image in images)
	shared = sorted(name for name, images_named in seen.items() if images_named > 1)
	if shared:
		raise SystemExit(f"more than one testbed image is named {', '.join(shared)}")
	unknown = sorted(frozenset(names) - seen.keys())
	if unknown:
		raise SystemExit(f"no testbed image is named {', '.join(unknown)}")
	chosen = tuple(image for image in images if (image.name in names if names else default(image)))
	if not chosen:
		raise SystemExit("no testbed image to run: check DCTR_FIXTURES and the pexplorer testdata")
	return chosen


def _checked_by_default(image: _Image, *, all_images: bool) -> bool:
	return all_images or image.name not in _SLOW_IMAGES


def _rtos_testbed(image: _Image) -> bool:
	return image.name in _RTOS_TESTBEDS


def _testbed_images() -> Iterable[_Image]:
	return chain(
		(
			_Image(build.name, build / "zephyr" / "zephyr.elf", build)
			for build in sorted(_fixtures().iterdir())
			if (build / "zephyr" / "zephyr.elf").is_file()
		),
		(
			_Image(elf.stem, elf, None)
			for elf in sorted(_PEXPLORER_TESTDATA.glob("*.elf"))
			if _is_elf(elf)
		),
	)


def _fixtures() -> Path:
	match os.environ.get("DCTR_FIXTURES"):
		case None:
			raise SystemExit("DCTR_FIXTURES is unset: run in `nix develop`, which exports it")
		case root:
			return Path(root)


def _is_elf(path: Path) -> bool:
	with path.open("rb") as stream:
		return stream.read(4) == b"\x7fELF"


def _dctr_commands(image: _Image) -> Iterable[_DctrCommand]:
	analyze = _DctrCommand(
		f"{image.name}.analyze", image.elf, ("analyze", "--json", str(image.elf))
	)
	match image.build:
		case None:
			return (analyze,)
		case build:
			return (
				analyze,
				_DctrCommand(
					f"{image.name}.stack",
					image.elf,
					("stack", str(build), "--elf", str(image.elf), "--json"),
				),
			)


async def _compared(
	trees: _Trees, command: _DctrCommand, limit: asyncio.Semaphore
) -> _OutputComparison:
	key = _key(command)
	base_output = await _kept(trees.base, command, key, limit)
	head_output = await _kept(trees.head, command, key, limit)
	return _OutputComparison(command, base_output == head_output)


async def _kept(
	tree: _Tree, command: _DctrCommand, key: str, limit: asyncio.Semaphore
) -> _Completed:
	record = _STORE / "outputs" / (tree.revision or "working-tree") / f"{command.output}.json"
	match _reusable(tree, record, key):
		case None:
			return _recorded(record, key, await _limited(limit, _dctr_output(tree, command)))
		case output:
			return output


def _key(command: _DctrCommand) -> str:
	return hashlib.sha256(
		b"\0".join(
			chain(
				map(str.encode, command.arguments),
				(
					sys.version.encode(),
					(_ROOT / "uv.lock").read_bytes(),
					hashlib.sha256(command.elf.read_bytes()).digest(),
				),
			)
		)
	).hexdigest()


def _reusable(tree: _Tree, record: Path, key: str) -> _Completed | None:
	if tree.revision is None or not record.is_file():
		return None
	try:
		kept = msgspec.json.decode(record.read_bytes(), type=_KeptOutput)
	except msgspec.DecodeError:
		return None
	return kept.output if kept.key == key else None


def _recorded(record: Path, key: str, output: _Completed) -> _Completed:
	if output.exit_status >= 0:
		record.parent.mkdir(parents=True, exist_ok=True)
		_replaced(record.with_name(f"{record.stem}.stdout"), output.stdout)
		_replaced(record.with_name(f"{record.stem}.stderr"), output.stderr)
		_replaced(record, msgspec.json.encode(_KeptOutput(key, output)))
	return output


def _replaced(path: Path, content: bytes) -> None:
	written = path.with_name(f"{path.name}.partial")
	written.write_bytes(content)
	written.replace(path)


async def _dctr_output(tree: _Tree, command: _DctrCommand) -> _Completed:
	process = await asyncio.create_subprocess_exec(
		sys.executable,
		"-c",
		_DCTR,
		*command.arguments,
		cwd=_ROOT,
		env=_environment(tree),
		stdout=asyncio.subprocess.PIPE,
		stderr=asyncio.subprocess.PIPE,
	)
	completed = await _completed(process)
	return _Completed(
		completed.exit_status,
		completed.stdout,
		completed.stderr.replace(os.fsencode(tree.source), b"<src>"),
	)


async def _completed(process: asyncio.subprocess.Process) -> _Completed:
	try:
		stdout, stderr = await process.communicate()
	except asyncio.CancelledError:
		process.kill()
		raise
	return _Completed(await process.wait(), stdout, stderr)


def _comparison_line(comparison: _OutputComparison) -> str:
	return f"{'identical' if comparison.identical else 'DIFFERS'} {comparison.command.output}"


async def _work_line(trees: _Trees, image: _Image, limit: asyncio.Semaphore) -> str:
	async with asyncio.TaskGroup() as both:
		base_run = both.create_task(_limited(limit, _measured(trees.base, "work", image.elf)))
		head_run = both.create_task(_limited(limit, _measured(trees.head, "work", image.elf)))
	base_work = msgspec.json.decode(base_run.result(), type=Work)
	head_work = msgspec.json.decode(head_run.result(), type=Work)
	return (
		f"{image.name}: transfers {trees.base.name} {base_work.transfers}, "
		f"{trees.head.name} {head_work.transfers}; function analyses {trees.base.name} "
		f"{base_work.function_analyses}, {trees.head.name} {head_work.function_analyses}"
	)


async def _pairs(
	trees: _Trees,
	image: _Image,
	pairs: int,
	load_below: float | None,
	stage: Literal["analysis", "load"],
) -> Collection[_Pair]:
	return [await _pair(trees, image, load_below, stage, index) for index in range(pairs)]


async def _pair(
	trees: _Trees,
	image: _Image,
	load_below: float | None,
	stage: Literal["analysis", "load"],
	index: int,
) -> _Pair:
	load_before = os.getloadavg()[0] if load_below is None else await _idle(load_below)
	order = (trees.base, trees.head) if index % 2 == 0 else (trees.head, trees.base)
	seconds = {tree.name: await _cpu_seconds(tree, image, stage) for tree in order}
	pair = _Pair(
		seconds[trees.base.name], seconds[trees.head.name], load_before, os.getloadavg()[0]
	)
	print(
		f"{image.name} pair {index + 1} ({' then '.join(seconds)}): {trees.base.name} "
		f"{pair.base_seconds:.2f} s, {trees.head.name} {pair.head_seconds:.2f} s, load "
		f"{pair.load_before:.2f} before and {pair.load_after:.2f} after",
		flush=True,
	)
	return pair


async def _idle(load_below: float, polls: int = 1) -> float:
	load = os.getloadavg()[0]
	if load < load_below:
		return load
	if polls >= _LOAD_POLLS:
		raise SystemExit(
			f"the 1-minute load average stayed at or above {load_below} for "
			f"{(_LOAD_POLLS - 1) * _LOAD_POLL_SECONDS} s; stop what loads the machine, or raise "
			"--load-below"
		)
	print(
		f"load {load:.2f}: waiting {_LOAD_POLL_SECONDS} s for it to fall below {load_below}",
		file=sys.stderr,
		flush=True,
	)
	await asyncio.sleep(_LOAD_POLL_SECONDS)
	return await _idle(load_below, polls + 1)


async def _cpu_seconds(tree: _Tree, image: _Image, stage: Literal["analysis", "load"]) -> float:
	measured = await _measured(tree, "cpu" if stage == "analysis" else "load", image.elf)
	return msgspec.json.decode(measured, type=CpuTime).cpu_seconds


def _cpu_summary(trees: _Trees, image: _Image, pairs: Collection[_Pair]) -> str:
	base_seconds = tuple(pair.base_seconds for pair in pairs)
	head_seconds = tuple(pair.head_seconds for pair in pairs)
	change = statistics.median(head_seconds) / statistics.median(base_seconds) - 1
	return (
		f"{image.name}: {trees.base.name} {_spread(base_seconds)}; {trees.head.name} "
		f"{_spread(head_seconds)}; median change {change:+.1%}"
	)


def _spread(seconds: Collection[float]) -> str:
	"""Describe the runs of one tree.

	>>> _spread((2.0, 1.0, 4.25))
	'min 1.00 s, median 2.00 s, max 4.25 s over 3 runs'
	"""
	return (
		f"min {min(seconds):.2f} s, median {statistics.median(seconds):.2f} s, "
		f"max {max(seconds):.2f} s over {len(seconds)} runs"
	)


async def _limited[T](limit: asyncio.Semaphore, work: Awaitable[T]) -> T:
	async with limit:
		return await work


async def _measured(tree: _Tree, kind: Literal["cpu", "load", "work"], elf: Path) -> bytes:
	return await _python(tree, "-m", "benchmarks.measure", kind, str(elf))


async def _python(tree: _Tree, *arguments: str) -> bytes:
	return await _output((sys.executable, *arguments), _environment(tree))


async def _git(*arguments: str) -> str:
	return (await _output(("git", *arguments))).decode()


async def _output(command: tuple[str, ...], environment: Mapping[str, str] | None = None) -> bytes:
	process = await asyncio.create_subprocess_exec(
		*command,
		cwd=_ROOT,
		env=environment,
		stdout=asyncio.subprocess.PIPE,
		stderr=asyncio.subprocess.PIPE,
	)
	completed = await _completed(process)
	if completed.exit_status:
		raise SystemExit(
			f"{shlex.join(command)} exited {completed.exit_status}:\n{completed.stderr.decode()}"
		)
	return completed.stdout


def _environment(tree: _Tree) -> dict[str, str]:
	return {**os.environ, "PYTHONPATH": str(tree.source)}


if __name__ == "__main__":
	app()
