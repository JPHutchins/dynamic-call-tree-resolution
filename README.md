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
--generate-report --report-type json`), on the same Zephyr CAN counter build that `dctr
stack` analyzes; both fed by GCC's `-fstack-usage`/`-fcallgraph-info` artifacts:

| | puncover | dctr |
|---|---|---|
| indirect calls detected | 0 (assembly-text regex) | 98 call sites |
| `poll_state_thread` worst case | 96 bytes | 340 bytes |
| `shell_readline` worst case | not reported (symbol match fails) | 1100 bytes |

puncover's indirect-call handling is an assembly-text detection flag, and its worst-case
traversal does not follow indirect edges; the 244-byte difference for
`poll_state_thread` corresponds to the `can_loopback_get_state` chain. pexplorer's
dynamic edges are likewise detected, with resolution deferred to a hand-maintained
config file. dctr resolves statically assigned function pointers exactly
(`dctr analyze`), reports per-site candidate sets (`dctr compare --pexplorer`), and
enumerates the runtime-assigned residue with member paths and signatures.

## References

- [pexplorer](https://paulwuertz.github.io/pexplorer/) — Paul Würtz's browser-based
  puncover reimplementation
- [puncover PR #157](https://github.com/HBehrens/puncover/pull/157) — GCC
  `-fcallgraph-info` VCG parsing
- AdaCore, *Compile-time stack requirements analysis with GCC* — introduced
  `-fstack-usage`/`-fcallgraph-info`
- [avstack](https://github.com/JPHutchins/avstack) — avr stack "worst case usage" tooling
