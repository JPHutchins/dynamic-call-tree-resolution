# dynamic-call-tree-resolution

Static resolution of indirect calls in embedded firmware ELF images, in service of
worst-case stack usage analysis.

## Problem

Worst-case stack analysis needs the call graph. C function pointers break it, and Zephyr
firmware is full of them: every driver API call is `device->api->fn(...)`, and every
`SYS_INIT` entry, ISR, thread entry point, and callback is an indirect call. Standard
tools report hundreds of "unresolved dynamic calls" and stop there.

## Strategy

1. **Static points-to (exact).** Every constant function pointer in a linked image is a
   baked data value or a relocation against a function symbol. Combined with DWARF
   structure layouts, `dev->api->open()` resolves to *exactly* the driver's `open`
   implementation.
2. **Init-system enumeration (exact).** Zephyr `SYS_INIT`/`DEVICE_DEFINE` entries are
   static data in linker sections — including device structs synthesized by the linker
   itself — so the init call graph is fully enumerable from the image.
3. **Candidate narrowing (sets).** The residue — runtime-assigned callbacks — is narrowed
   by matching the function-pointer type's DWARF signature and by value-set analysis
   rooted at the init functions.

Worst-case stack usage follows from the resolved call graph combined with GCC's
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
z_main_thread.base.timeout.fn: <unresolved>
_thread_dummy.base.timeout.fn: <unresolved>
_stdout_hook: <unresolved>
...
char_out@0x118: arch_printk_char_out
...
console_out@0x956: uart_stellaris_poll_out
console_out@0x960: <unresolved>
```

- `__init_*.init_fn` lines are Zephyr `SYS_INIT` entries, enumerated exactly from their
  linker sections; `__device_dts_ord_22.ops.init` is the linker-synthesized device struct's
  init function.
- The `<unresolved>` lines are the runtime-assigned residue — thread timeout callbacks and
  the console output hook — enumerated with member paths for manual review.
- `console_out@0x956: uart_stellaris_poll_out` is an indirect call site (`blx r3`) resolved
  through the `device->api` chain to its single target — the slot layer sees what the site
  alone cannot.

## Tool comparison

[puncover](https://github.com/HBehrens/puncover) 0.8.0, non-interactive report mode
(`puncover --elf <exe> --build_dir <build> --gcc-tools-base /usr/bin --non-interactive
--generate-report --report-type json`), on the Zephyr CAN counter build whose linked
executable and `-fstack-usage`/`-fcallgraph-info` artifacts are committed at
`tests/fixtures/counter-su` (`dctr stack tests/fixtures/counter-su` and `dctr
summary tests/fixtures/counter-su tests/fixtures/counter-su/zephyr/zephyr.exe`;
the numbers below are pinned by `tests/test_counter_fixture.py`):

| | puncover | dctr |
|---|---|---|
| indirect calls detected | 0 (assembly-text regex) | 98 call sites |
| `poll_state_thread` worst case | 96 bytes | 460 bytes (static-only; 996 with indirect expansion) |
| `shell_readline` worst case | not reported (symbol match fails) | 2108 bytes (upper bound; 1760 without indirect expansion) |
| slots resolved/unresolved/total | — | 75/78/153 |
| call sites resolved/exact/total | — | 6/6/98 |

puncover's indirect-call handling is an assembly-text detection flag, and its reported
worst case for `poll_state_thread` contains only the function itself, omitting the
static callee depth the same artifacts yield (details below). pexplorer's dynamic
edges are likewise detected, with resolution deferred to a hand-maintained config
file. dctr resolves statically assigned function pointers exactly (`dctr analyze`),
reports per-site candidate sets (`dctr compare --pexplorer`), and enumerates the
runtime-assigned residue with member paths and signatures.

<details>
<summary>poll_state_thread trees — puncover 0.8.0 report vs dctr</summary>

puncover's report (`stack_report.poll_state_thread`, `call_stack` in full):

```
poll_state_thread (96)
```

dctr's deepest path over static `.ci` edges (frame bytes; cumulative in
parentheses):

```
poll_state_thread (96, 96)
└── k_sleep_ticks (32, 128)                  [static]
    └── z_impl_k_sleep_ticks (64, 192)       [static]
        └── z_impl_k_yield (4, 196)          [static]
            └── z_sched_yield (48, 244)      [static]
                └── z_time_slice_reset (16, 260)  [static]
                    └── slice_reset (64, 324)     [static]
                        └── z_add_timeout (80, 404)  [static]
                            └── sys_clock_set_timeout (8, 412)  [static]
                                └── timer_core_arm (48, 460)    [static]
                                    └── hwtimer_set_tick_one_shot (0, 460)  [static]
```

Every edge on this path is a static `.ci` edge; the frames are the `.su` record
bytes, so the 364-byte difference is static callee depth that puncover's report
omits. The function's one indirect call site is resolved exactly by dctr to
`can_loopback_get_state` (8 bytes) — exact, but on a separate branch, not the
deepest one. With indirect expansion, dctr's upper bound for this entry is 996
bytes: the expansion unions every resolved target into each indirect edge, so the
deepest expanded path runs through the fallback targets rather than the all-static
path shown above.

</details>

<details>
<summary>shell_readline deepest path — the one genuinely indirect edge</summary>

dctr's deepest path over the committed artifacts (frame bytes; cumulative in
parentheses):

```
shell_readline (96, 96)
└── state_collect (96, 192)                      [static]
    └── tab_handle (480, 672)                    [static]
        └── z_shell_op_char_insert (64, 736)     [static]
            └── data_insert (64, 800)            [static]
                └── reprint_from_cursor (80, 880)     [static]
                    └── z_shell_fprintf (32, 912)     [static]
                        └── z_shell_vfprintf (8, 920) [static]
                            └── z_shell_print (80, 1000)  [static]
                                └── z_shell_vt100_colors_restore (32, 1032)  [static]
                                    └── z_shell_vt100_color_set (32, 1064)   [static]
                                        └── z_shell_raw_fprintf (32, 1096)   [static]
                                            └── z_shell_fprintf_fmt (32, 1128)  [static]
                                                └── cbvprintf (48, 1176)     [static]
                                                    └── z_cbvprintf_impl (160, 1336)  [static]
                                                        └── outs (64, 1400)  [static]
                                                            └── z_shell_print_stream (4, 1404)  [indirect]
                                                                └── z_shell_write (80, 1484)  [static]
                                                                    ... (kernel wait, fatal, scheduler frames — all [static])
                                                                    └── timer_core_arm (48, 2108)  [static]
                                                                        └── hwtimer_set_tick_one_shot (0, 2108)  [static]
```

The only indirect edge on the path is `outs → z_shell_print_stream`: the shell's
print-stream hook, which dctr resolves to a single candidate. dctr reports 2108 with
indirect expansion and 1760 over static `.ci` edges alone.

</details>

## References

- [pexplorer](https://paulwuertz.github.io/pexplorer/) — Paul Würtz's browser-based
  puncover reimplementation
- [puncover PR #157](https://github.com/HBehrens/puncover/pull/157) — GCC
  `-fcallgraph-info` VCG parsing
- AdaCore, *Compile-time stack requirements analysis with GCC* — introduced
  `-fstack-usage`/`-fcallgraph-info`
- [avstack](https://github.com/JPHutchins/avstack) — avr stack "worst case usage" tooling
