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
   over every function's machine code, seeded from observed direct calls, and by
   matching the function-pointer type's DWARF signature. The narrowed sets are
   refinements, not over-approximations.

Stack depths combine the resulting call graph with GCC's
`-fstack-usage`/`-fcallgraph-info` build artifacts.

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
char_out@0x118: console_out, arch_printk_char_out
...
_isr_wrapper@0x884: z_irq_spurious
z_impl_zephyr_fputc@0x8a6: console_out
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
zephyr.exe                                     EM_386           734 121/113/234     26/14/98
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
shell_readline: 1760 bytes recursive dynamic unmeasured: 1
...
poll_state_thread: 460 bytes dynamic unmeasured: 1
...
```

```console
$ dctr stack tests/fixtures/counter-su --elf tests/fixtures/counter-su/zephyr/zephyr.exe
resolved slots: 121 | indirect call sites: 100
shell_readline: 2108 bytes recursive dynamic unmeasured: 1
...
poll_state_thread: 1012 bytes recursive dynamic
...
```

- A *resolved* site or slot has at least one candidate, and an *exact* site has
  exactly one, in the image as linked.
- `compare` counts the indirect instructions it extracts; `stack --elf` counts the
  `__indirect_call` edges in the `.ci` files.
- `stack --elf` expands every indirect edge to its site candidates plus the fallback
  union of all resolved slot targets. `outs` has no candidates, so its edge is the
  fallback alone.
- None of these depths is a worst-case bound. `recursive` (a cycle was broken) and
  `dynamic` (a frame uses `alloca` or a VLA) describe any branch of the entry's
  subtree; `unmeasured: N` counts frames on the deepest path that have no `.su` record
  and count as 0 bytes.
- `shell_readline` is not in the linked executable: the linker discarded it, but its
  `.ci` graph survives, and entry points come from the `.ci` graph.

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
- Value-set analysis per-site sets are refinements, not over-approximations: the
  analysis drops unknown values and does not model every write ([#61]), and its
  control-flow graph has gaps ([#66]).
- Signature narrowing compares DWARF signatures for equality: a cast defeats it, and
  functions whose DWARF name renders as `<anonymous>` never match ([#63]).
- The fallback for sites with no candidates is the union of data-slot targets; it
  misses functions whose address appears only in code ([#62]).
- Code without `.ci` records (assembly, `native_sim` host code) is absent from the
  stack call graph ([#78]).

### Stack depths

- A depth is the deepest path the search found over the call graph; it is a bound
  only if that graph is complete, and today it is not (above).
- Expanding indirect edges (`stack --elf`) can report less than the subset-only
  expansion used in the tests ([#58]).
- `recursive`, `dynamic`, and `unmeasured: N` results are not bounded at all ([#64]).
- Entry points come from `.ci`, including functions the linker discarded ([#64], [#78]).
- Interrupt, exception, context-switch, and FPU stacking are not modeled.

### Residue

`<unresolved>` slots are enumerated from DWARF-typed data objects. Array members,
pointer-to-pointer members, anonymous structs, location lists ([#17]), and heap or stack
storage are not enumerated ([#69]).

## Development

```sh
nix develop            # uv, the host C compiler, jphfmt, nixfmt, git-lfs
uv run camas           # the checks CI runs
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
[#61]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/61
[#62]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/62
[#63]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/63
[#64]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/64
[#66]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/66
[#67]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/67
[#69]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/69
[#71]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/71
[#74]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/74
[#78]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/78
