# dynamic-call-tree-resolution

Static resolution of indirect calls in embedded firmware ELF images, in service of
worst-case stack usage analysis.

## Problem

Worst-case stack analysis needs the call graph. C function pointers break it, and Zephyr
firmware is full of them: every driver API call is `device->api->fn(...)`, and every
`SYS_INIT` entry, ISR, thread entry point, and callback is an indirect call. Standard
tools report hundreds of "unresolved dynamic calls" and stop there.

## Strategy

Three layers, in order of strength:

1. **Static points-to (exact).** Every constant function pointer in a linked image is a
   baked data value or a relocation against a function symbol. Combined with DWARF
   structure layouts, `dev->api->open()` resolves to *exactly* the driver's `open`
   implementation. This layer is implemented: `dctr analyze firmware.elf`.
2. **Init-system enumeration (exact).** Zephyr `SYS_INIT`/`DEVICE_DEFINE` entries are
   static data in linker sections; the init call graph is fully enumerable from the image.
3. **Candidate narrowing (sets).** The residue — runtime-assigned callbacks — is narrowed
   by matching the function-pointer type's DWARF signature and by value-set analysis
   rooted at the init functions.

## Layout

- `src/dynamic_call_tree_resolution/` — the library (typed facade over pyelftools;
  type stubs in `typings/`)
- `tests/` — pytest + doctests, fixtures compiled from `tests/fixtures/*.c` at test time
- `testbeds/` — real firmware: mainline `zephyr`, `zmk`, `zswatch`
- `references/` — `camas` (task-runner reference), Paul Würtz's `pexplorer`

## Tasks

`camas` is the single task runner — the same definitions drive local work and CI.

| task | what it runs |
| --- | --- |
| `camas` (default = `all`) | fix, then the full type-check ladder (mypy, pyright, ty, zuban, pyrefly) + 100% coverage |
| `camas check` | read-only: format, lint, type checks, tests (the GitHub task) |
| `camas gate` | as `check`, with the coverage gate |
| `camas matrix` | per-Python-version matrix from `.python-version` |
| `dctr analyze firmware.elf --json` | resolve and print assignments |

## References

- [pexplorer](https://paulwuertz.github.io/pexplorer/) — Paul Würtz's browser-based
  puncover reimplementation
- [puncover PR #157](https://github.com/HBehrens/puncover/pull/157) — GCC
  `-fcallgraph-info` VCG parsing
- AdaCore, *Compile-time stack requirements analysis with GCC* — introduced
  `-fstack-usage`/`-fcallgraph-info`
- [avstack](https://github.com/JPHutchins/avstack) — avr stack "worst case usage" tooling
