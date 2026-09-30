# dynamic-call-tree-resolution

Static resolution of indirect calls in embedded firmware ELF images, in service of
worst-case stack usage analysis.

> [!CAUTION]
> dctr's stack depths are **not** sound worst-case bounds, and its call-target sets are
> **not** guaranteed complete. Do not use either for safety or certification
> sign-off. See [Limitations and soundness](#limitations-and-soundness).

## Problem

Worst-case stack analysis needs the call graph. C function pointers break it, and Zephyr
firmware is full of them: every driver API call is `device->api->fn(...)`, and every
`SYS_INIT` entry, ISR, thread entry point, and callback is an indirect call. GCC's
`-fcallgraph-info` records each of them as an edge to the placeholder `__indirect_call`,
with no target.

## Strategy

1. **Static points-to.** A function pointer stored in initialized data is either a baked
   value or a relocation against a function symbol. Combined with DWARF structure
   layouts, every such slot is read from the image as linked. A slot in a writable
   section holds its initializer, which code may overwrite at runtime.
2. **Init-system enumeration.** Zephyr `SYS_INIT`/`DEVICE_DEFINE` entries are static
   data that the linker places in dedicated sections, so their slots are enumerated
   from the image. The dispatch sites that call through them (`z_sys_init_run_level`,
   `do_device_init`) are not resolved.
3. **Candidate narrowing.** Indirect call sites are narrowed by a value-set analysis
   over every function's machine code, seeded from observed direct calls. The narrowed
   sets are refinements, not over-approximations. With `--narrow-by-signature`, a site
   whose slot the image leaves unset also narrows to the functions of the slot's DWARF
   signature.

Stack depths combine the resulting call graph with GCC's
`-fstack-usage`/`-fcallgraph-info` build artifacts. Each indirect call also expands to
the fallback: every function whose address the image stores, or a non-branch
instruction computes.

## Usage

On Zephyr's `hello_world` for `qemu_cortex_m3` (`tests/fixtures/hello_zephyr_qemu_cortex_m3.elf`):

```console
$ dctr analyze tests/fixtures/hello_zephyr_qemu_cortex_m3.elf
...
__init_uart_stellaris_init.init_fn: uart_stellaris_init
...
__device_dts_ord_22.ops.init: uart_stellaris_init
...
uart_stellaris_driver_api.configure: <unresolved>
...
z_main_thread.base.timeout.fn: <unresolved>
_thread_dummy.base.timeout.fn: <unresolved>
...
_stdout_hook: <unresolved>
...
char_out@0x118: <unresolved>
...
_isr_wrapper@0x884: z_irq_spurious
z_impl_zephyr_fputc@0x8a6: <unresolved>
...
console_out@0x956: uart_stellaris_poll_out
console_out@0x960: uart_stellaris_poll_out
z_sys_init_run_level@0xcd0: <unresolved>
...
do_device_init@0x1dca: <unresolved>
```

- `__init_*.init_fn` lines are Zephyr `SYS_INIT` entries, read from their linker
  sections; `__device_dts_ord_22.ops.init` is the device struct's init function.
- `<unresolved>` means the slot holds no function address in the image as linked. That
  covers slots assigned at runtime (the thread timeout callbacks and `_stdout_hook`
  above), constant `NULL` members (`uart_stellaris_driver_api.configure`), and union
  arms that are not function pointers; they are listed with member paths for manual
  review.
- `char_out@0x118` and `z_impl_zephyr_fputc@0x8a6` call through `_char_out`, a function
  pointer in RAM. The image has stores whose address the analysis cannot compute, so any
  writable slot may hold anything: `analyze` reports these sites unresolved, and
  `stack --elf` expands them to every address-taken function.
- `console_out@0x956` and `console_out@0x960` are indirect call sites (`blx r3`) whose
  value-set analysis through the `device->api` chain yields one candidate.
- `_isr_wrapper@0x884` resolves to `z_irq_spurious` through the indexed load over
  `_sw_isr_table`, which is read-only in this image: every entry's handler points there.

## Counter build

The Zephyr CAN counter sample for `native_sim` (an x86 host executable), with its
`-fstack-usage`/`-fcallgraph-info` artifacts, at `tests/fixtures/counter-su`:

```console
$ dctr compare tests/fixtures/counter-su/zephyr/zephyr.exe
elf                                            machine    functions slots r/u/t  sites r/e/t
zephyr.exe                                     EM_386           734 121/113/234       9/8/98
```

```console
$ dctr analyze tests/fixtures/counter-su/zephyr/zephyr.exe
...
poll_state_thread@0x8049dad: can_loopback_get_state
...
outs@0x804a8fa: <unresolved>
...
```

```console
$ dctr stack tests/fixtures/counter-su
shell_readline: unbounded, at least 1760 bytes (recursion: 4, unmeasured: 22, unresolved: 10)
...
poll_state_thread: unbounded, at least 460 bytes (unmeasured: 7, unresolved: 1)
...
```

```console
$ dctr stack tests/fixtures/counter-su --elf tests/fixtures/counter-su/zephyr/zephyr.exe
resolved slots: 121 | indirect call sites: 100 | not in the image: 222
cmd_can_send: unbounded, at least 3196 bytes (recursion: 100, unmeasured: 130)
...
poll_state_thread: unbounded, at least 1940 bytes (recursion: 100, unmeasured: 129)
...
```

- A *resolved* site or slot has at least one candidate, and an *exact* site has
  exactly one, in the image as linked.
- `compare` counts the indirect instructions it extracts; `stack --elf` counts the
  `__indirect_call` edges in the `.ci` files.
- `stack --elf` expands every indirect edge to its site candidates plus the fallback,
  every address-taken function. `outs` has no candidates, so its edge is the fallback
  alone.
- An `unbounded` entry's number is only a lower bound. The counts name what breaks the
  bound anywhere in the entry's subtree:
  - `recursion`: functions on a cycle;
  - `unmeasured`: functions with no `.su` record, counted as 0 bytes;
  - `dynamic`: frames GCC could not bound (`alloca` or a VLA);
  - `unresolved`: callers of an indirect call with no candidates.
- A plain `N bytes` bounds every path of the call graph as given. That graph is still
  incomplete ([#66], [#96], [#113]).
- Entry points come from the `.ci` graph, which also records functions the linker
  discarded, such as `shell_readline`. With an ELF, entries missing from the image are
  dropped and counted as `not in the image`; without one, every `.ci` entry is listed.
- `stack --json` carries the full function names behind each count.

## Related tools

- [puncover](https://github.com/HBehrens/puncover) 0.8.0 reads `-fstack-usage` but not
  `-fcallgraph-info`; it recovers calls and indirect calls from disassembly with
  ARM-only regexes (`BLX\s+(\w+)$` in `gcc_tools.py`), so on an x86 build such as the
  counter it sees no calls at all. A like-for-like comparison on an ARM build is
  tracked in [#74].
- [pexplorer](https://paulwuertz.github.io/pexplorer/) detects dynamic edges and
  defers their resolution to a hand-maintained config file. `dctr compare --pexplorer`
  joins its report per function, comparing pexplorer's dynamic-call count with dctr's
  resolved and exact site counts; the per-site sets come from `dctr analyze`.

## Limitations and soundness

Known soundness bugs are tracked under [#59]. Until each is closed, the claim it
contradicts does not hold.

### Supported inputs

- A linked ELF executable with DWARF; relocatable objects are rejected. Stack depths
  also need GCC's `-fstack-usage` (`.su`) and `-fcallgraph-info` (`.ci`) artifacts.
- `EM_ARM`, `EM_386`, and `EM_X86_64`; other machines are rejected. `EM_ARM` code is
  decoded as Thumb only, with no A32 detection.
- Zephyr is the only RTOS modeled, and its knowledge is not isolated ([#71]).

### Call targets

- A slot's value is its value in the image as linked. Writable-section slots report
  their initializer, which runtime code may replace ([#67]).
- *Exact* means one candidate in the image as linked, not the only function the site
  can call at runtime.
- A store whose address the analysis cannot compute makes every writable address
  unknown, so a site that reads its target from RAM is unresolved unless its own path
  wrote that target.
- Value-set analysis per-site sets are refinements, not over-approximations: its
  control-flow graph has gaps ([#66]), and it does not model x86 sub-registers or a few
  kinds of write ([#113]).
- `--narrow-by-signature` compares DWARF signatures for equality, so a cast defeats
  it. It is off by default.
- The fallback holds every function address the image stores, or that one instruction
  or a `movw`/`movt` pair in a function symbol computes. A function pointer built by
  other arithmetic, or in code outside every function symbol, is missed ([#96]).
- Code without `.ci` records (assembly, `native_sim` host code) is absent from the
  stack call graph ([#78]).

### Stack depths

- A depth is the deepest path the search found over the call graph; it is a bound
  only if that graph is complete, and today it is not (above).
- Expanding indirect edges (`stack --elf`) can report less than the subset-only
  expansion used in the tests ([#58]).
- Without an ELF, entry points come from `.ci`, including functions the linker
  discarded ([#78]).
- Interrupt, exception, context-switch, and FPU stacking are not modeled.

### Residue

`<unresolved>` slots are enumerated from DWARF-typed data objects. Array members,
pointer-to-pointer members, anonymous structs, location lists ([#17]), and heap or stack
storage are not enumerated ([#69]).

## Development

```sh
nix develop            # uv, the host and arm-none-eabi C compilers, QEMU, jphfmt, nixfmt, git-lfs
uv run camas           # the checks CI runs
uv run camas check_fast   # the same checks, skipping the tests marked image
uv run camas matrix    # the same checks on each interpreter in .python-version
```

uv manages the Python interpreters named in `.python-version`. Tools started from the
shell, such as an editor or `camas mcp`, inherit its toolchain.

## References

- [pexplorer](https://paulwuertz.github.io/pexplorer/) — Paul Würtz's browser-based
  puncover reimplementation
- [puncover PR #157](https://github.com/HBehrens/puncover/pull/157) — GCC
  `-fcallgraph-info` VCG parsing
- AdaCore, *Compile-time stack requirements analysis with GCC* — introduced
  `-fstack-usage`/`-fcallgraph-info`
- [avstack](https://github.com/JPHutchins/avstack) — avr stack "worst case usage" tooling

[#17]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/17
[#58]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/58
[#59]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/59
[#66]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/66
[#67]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/67
[#69]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/69
[#71]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/71
[#74]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/74
[#78]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/78
[#96]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/96
[#113]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/113
