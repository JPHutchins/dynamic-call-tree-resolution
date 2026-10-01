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
   layouts, every such slot is read from the image as linked. A slot in read-only memory
   keeps that value. A slot in writable memory holds its initializer plus every value
   the value-set analysis sees the program store there, and is unresolved when a store's
   address or value is unknown.
2. **Init-system enumeration.** Zephyr `SYS_INIT`/`DEVICE_DEFINE` entries are static
   data that the linker places in dedicated sections, so their slots are enumerated
   from the image. The dispatch sites that call through them (`z_sys_init_run_level`,
   `do_device_init`) are not resolved.
3. **Candidate narrowing.** Indirect call sites are narrowed by a value-set analysis
   over every function's machine code, seeded from observed direct calls. The narrowed
   sets are refinements, not over-approximations. With `--narrow-by-signature`, a site
   whose slot the image leaves unset also narrows to the functions of the slot's DWARF
   signature. Each RTOS static thread is also analyzed on its own, started from its
   record.

Stack depths combine the resulting call graph with GCC's
`-fstack-usage`/`-fcallgraph-info` build artifacts. Each indirect call also expands to
the fallback: every function whose address the image stores, or a non-branch
instruction computes. The exception is a site inside a thread's tree that the thread's
own analysis resolved.

## Usage

The examples analyze Zephyr builds from `nix build .#fixtures`, which `nix develop`
exports as `$DCTR_FIXTURES` (see [Development](#development)).

On Zephyr's `hello_world` for `qemu_cortex_m3` (`$DCTR_FIXTURES/hello/zephyr/zephyr.elf`):

```console
$ dctr analyze $DCTR_FIXTURES/hello/zephyr/zephyr.elf
rtos: zephyr (detected: z_thread_entry, struct _static_thread_data)
...
__init_uart_stellaris_init.init_fn: uart_stellaris_init
...
__device_dts_ord_22.ops.init: uart_stellaris_init
...
uart_stellaris_driver_api.configure: <null>
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

- `rtos:` names the RTOS model and what identified it in the image. `--rtos auto`, the
  default, detects Zephyr; `--rtos none` models no RTOS. Each Zephyr static thread
  (`K_THREAD_DEFINE`) prints as `thread NAME: ENTRY`, and `hello_world` defines none.
  An entry is *seeded* when its thread's read-only record is the only place the image
  holds its address: the analysis then starts it with the record's `p1`..`p3` instead
  of unknown arguments. This assumes the kernel's static-thread start is the only code
  that reads the records. `analyze` prints this whole-image analysis; `stack --elf`
  also analyzes each static thread on its own (below).
- `__init_*.init_fn` lines are Zephyr `SYS_INIT` entries, read from their linker
  sections; `__device_dts_ord_22.ops.init` is the device struct's init function.
- `<unresolved>` means the analysis has no function address for a writable slot. That
  covers slots assigned at runtime (the thread timeout callbacks and `_stdout_hook`
  above) and writable slots whose stores are unknown (`_char_out` and `__stdout.put`).
- `<null>` marks a read-only slot that holds NULL, such as an optional driver operation
  the driver leaves out (`uart_stellaris_driver_api.configure`): it can never hold a
  function. `<not a function>` marks a read-only slot that holds something else, such as
  a union arm that overlaps a member that is not a function pointer.
- All three are listed with member paths for manual review, and all three count as
  unresolved in `compare` and `summary`.
- `<not enumerated: reason>` marks an object or member that may hold or lead to a
  function pointer, but whose slots are not listed:
  - a pointer to a pointer, or to an array, whose target can hold a pointer;
  - an array of unknown size;
  - an object without a type that is writable, or only partly holds function addresses.

  Heap and stack slots, and variables whose location is a location list ([#17]), are
  not listed at all.
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
`-fstack-usage`/`-fcallgraph-info` artifacts, at `$DCTR_FIXTURES/counter-su`:

```console
$ dctr compare $DCTR_FIXTURES/counter-su/zephyr/zephyr.exe
elf                                            machine    functions slots r/u/t  sites r/e/t
zephyr.exe                                     EM_386           729 106/144/250       8/7/97
```

```console
$ dctr analyze $DCTR_FIXTURES/counter-su/zephyr/zephyr.exe
...
poll_state_thread@0x8049dc1: can_loopback_get_state
...
z_shell_write@0x804f506: <unresolved>
...
```

```console
$ dctr stack $DCTR_FIXTURES/counter-su
shell_readline: unbounded, at least 1728 bytes (recursion: 4, unmeasured: 21, unresolved: 9)
...
poll_state_thread: unbounded, at least 428 bytes (unmeasured: 6, unresolved: 1)
...
```

```console
$ dctr stack $DCTR_FIXTURES/counter-su --elf $DCTR_FIXTURES/counter-su/zephyr/zephyr.exe
resolved slots: 106 | indirect call sites: 101 | not in the image: 286 | rtos: zephyr
gpio_emul_port_set_masked_raw: unbounded, at least 6288 bytes (recursion: 222, unmeasured: 153)
...
poll_state_thread: unbounded, at least 6192 bytes (recursion: 222, unmeasured: 153)
...
```

```console
$ dctr stack $DCTR_FIXTURES/counter-su --path poll_state_thread
poll_state_thread: unbounded, at least 428 bytes (unmeasured: 6, unresolved: 1)
poll_state_thread +80 = 80 bytes (unresolved)
k_sleep_ticks +32 = 112 bytes via static
z_impl_k_sleep_ticks +64 = 176 bytes via static
z_impl_k_yield +4 = 180 bytes via static
z_sched_yield +32 = 212 bytes via static
z_time_slice_reset +16 = 228 bytes via static
slice_reset +64 = 292 bytes via static
z_add_timeout +80 = 372 bytes via static
sys_clock_set_timeout +8 = 380 bytes via static
timer_core_arm +48 = 428 bytes via static
hwtimer_set_tick_one_shot +0 = 428 bytes via static (unmeasured)
```

```console
$ dctr stack $DCTR_FIXTURES/counter-su --elf $DCTR_FIXTURES/counter-su/zephyr/zephyr.exe --path poll_state_thread
resolved slots: 106 | indirect call sites: 101 | not in the image: 286 | rtos: zephyr
poll_state_thread: unbounded, at least 6192 bytes (recursion: 222, unmeasured: 153)
poll_state_thread +80 = 80 bytes (recursion)
can_msgq_put +32 = 112 bytes via indirect: fallback (recursion)
z_impl_k_msgq_put +8 = 120 bytes via static (recursion)
...
tx_thread +112 = 624 bytes via indirect: fallback (recursion)
...
timer_core_arm +48 = 6192 bytes via static
hwtimer_set_tick_one_shot +0 = 6192 bytes via static (unmeasured)
```

```console
$ dctr stack $DCTR_FIXTURES/counter-su --path shell_readline
shell_readline: unbounded, at least 1728 bytes (recursion: 4, unmeasured: 21, unresolved: 9)
shell_readline +80 = 80 bytes
state_collect +96 = 176 bytes via static (unresolved)
execute +432 = 608 bytes via static (unresolved)
...
encode_uint +112 = 1728 bytes via static
__ctype_b_loc +0 = 1728 bytes via static (unmeasured)
```

- A *resolved* site or slot has at least one candidate, and an *exact* site has
  exactly one, in the image as linked.
- `compare` counts the indirect instructions it extracts; `stack --elf` counts the
  `__indirect_call` edges in the `.ci` files. The `stack --elf` header ends with the
  RTOS model when one is detected.
- `stack --elf` expands every indirect edge to its site candidates plus the fallback,
  every address-taken function, except inside a thread's tree (below). `z_shell_write`
  has no candidates, so its edge is the fallback alone, and every function it reaches
  that way is `indirect: fallback`.
- An `unbounded` entry's number is only a lower bound. The counts name what breaks the
  bound anywhere in the entry's subtree:
  - `recursion`: functions on a cycle;
  - `unmeasured`: functions with no `.su` record, counted as empty frames;
  - `dynamic`: frames GCC could not bound (`alloca` or a VLA);
  - `unresolved`: callers of an indirect call with no candidates.
- A plain `N bytes` bounds every path of the call graph as given. That graph is still
  incomplete ([#96], [#113]).
- Entry points come from the `.ci` graph, which also records functions the linker
  discarded, such as `shell_readline`. With an ELF, calls from discarded functions are
  ignored, so a function only they call, such as `work_queue_main`, is an entry, and
  discarded entries are dropped and counted as `not in the image`. Without one, every
  `.ci` entry is listed.
- With an ELF and an RTOS model, each static thread is an entry named for its thread,
  such as `thermal_tid`. It is `z_thread_entry`'s frame and calls, with the indirect call
  that starts the thread going to that thread's entry alone. The entry function is then
  no longer an entry of its own.
- Below that, a thread's tree comes from an analysis of the thread alone, started from
  its record. A site that analysis tracked to known functions expands to those functions
  without the fallback, provided it decoded as many indirect calls in the function as
  `.ci` records. So two threads that share an entry but pass it different bindings get
  different trees. Any other site expands as it does everywhere else, and below it the
  thread's tree is the image's.
- `stack --json` carries the full function names behind each count.
- `stack --path ENTRY` prints that entry alone, then its deepest path, one function per
  line: its frame, the running total, the edge it is called through (`static`,
  `indirect: candidate`, `indirect: fallback`, or `thread record`), and what its own
  frame, calls, or cycle add to an unbounded depth. The last total is the entry's depth.

## Related tools

- [puncover](https://github.com/HBehrens/puncover) 0.8.0 reads `-fstack-usage` but not
  `-fcallgraph-info`; it recovers calls and indirect calls from disassembly with
  ARM-only regexes (`BLX\s+(\w+)$` in `gcc_tools.py`), so on an x86 build such as the
  counter it sees no calls at all. A like-for-like comparison on an ARM build is
  tracked in [#131].
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
  decoded as A32 inside the spans of `$a` mapping symbols and as Thumb elsewhere.
- Zephyr is the only RTOS modeled, and only its static threads are. A thread created
  with `k_thread_create` starts its entry with unknown arguments, and the indirect call
  in `z_thread_entry` that starts every thread falls back to every address-taken
  function ([#146]).

### Call targets

- A read-only slot's value is its value in the image as linked. A writable slot's
  candidates are its initializer and the values stored to it, so a store whose address
  or value is unknown leaves it unresolved.
- *Exact* means one candidate in the image as linked, not the only function the site
  can call at runtime.
- A store whose address the analysis cannot compute makes every writable address
  unknown, so a site that reads its target from RAM is unresolved unless its own path
  wrote that target.
- Value-set analysis per-site sets are refinements, not over-approximations: its
  control-flow graph misses x86 `notrack` switches, and it does not model x86
  sub-registers or a few kinds of write ([#113]).
- Inside a thread's tree, a site the thread's analysis resolved drops the fallback, so a
  gap in that analysis drops a target there.
- Zephyr's I2C and SPI emulators find their target in a list that init code builds in
  RAM, so their sites stay unresolved, and two threads' trees rejoin below them ([#151],
  [#159]).
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
- In an entry that reaches a large cycle, `at least` is the depth of one path through
  the cycle that a depth-first walk finds, not the deepest. More edges, such as the
  fallback's, can print a smaller number.
- Without an ELF, entry points come from `.ci`, including functions the linker
  discarded, and a function only they call is not an entry ([#78]).
- Interrupt, exception, context-switch, and FPU stacking are not modeled, so a
  thread's depth leaves out the frames an interrupt pushes onto its stack.

### Residue

Unresolved slots are enumerated from DWARF-typed data objects, including array members
and anonymous structs. The kinds marked `<not enumerated>` are listed without slots.
Location lists ([#17]) and heap or stack storage are not enumerated at all.

## Development

```sh
nix develop            # uv, the host and arm-none-eabi C compilers, QEMU, jphfmt, nixfmt, git-lfs
uv run camas           # the checks CI runs
uv run camas check_fast   # the same checks, skipping the tests marked image
uv run camas matrix    # the same checks on each interpreter in .python-version
```

uv manages the Python interpreters named in `.python-version`. Tools started from the
shell, such as an editor or `camas mcp`, inherit its toolchain.

The Zephyr testbeds build in their own shell, which adds the Zephyr SDK, dtc and the
multilib host gcc that `native_sim` links with. Its host gcc is not the one the checks
use, so run the checks from `nix develop`.

```sh
nix develop .#testbeds
uv run camas testbeds_init            # the west workspace testbeds/manifest/west.yml pins
uv run camas testbeds --NAME=hello    # one testbed, into .camas/build/hello
uv run camas testbeds_lock            # after editing testbeds/manifest/west.yml
```

`nix build .#fixtures` builds the same testbeds in the Nix sandbox, from the projects
`testbeds/west2nix.toml` locks. `nix develop` exports that build as `$DCTR_FIXTURES`,
which the tests marked image and the examples above read, so the first `nix develop` on
a machine builds it.

## References

- [pexplorer](https://paulwuertz.github.io/pexplorer/) — Paul Würtz's browser-based
  puncover reimplementation
- [puncover PR #157](https://github.com/HBehrens/puncover/pull/157) — GCC
  `-fcallgraph-info` VCG parsing
- AdaCore, *Compile-time stack requirements analysis with GCC* — introduced
  `-fstack-usage`/`-fcallgraph-info`
- [avstack](https://github.com/JPHutchins/avstack) — avr stack "worst case usage" tooling

[#17]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/17
[#59]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/59
[#78]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/78
[#96]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/96
[#113]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/113
[#131]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/131
[#146]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/146
[#151]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/151
[#159]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/159
