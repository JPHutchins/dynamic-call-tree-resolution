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

## References

- [pexplorer](https://paulwuertz.github.io/pexplorer/) — Paul Würtz's browser-based
  puncover reimplementation
- [puncover PR #157](https://github.com/HBehrens/puncover/pull/157) — GCC
  `-fcallgraph-info` VCG parsing
- AdaCore, *Compile-time stack requirements analysis with GCC* — introduced
  `-fstack-usage`/`-fcallgraph-info`
- [avstack](https://github.com/JPHutchins/avstack) — avr stack "worst case usage" tooling
