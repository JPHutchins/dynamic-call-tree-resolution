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

## References

- [pexplorer](https://paulwuertz.github.io/pexplorer/) — Paul Würtz's browser-based
  puncover reimplementation
- [puncover PR #157](https://github.com/HBehrens/puncover/pull/157) — GCC
  `-fcallgraph-info` VCG parsing
- AdaCore, *Compile-time stack requirements analysis with GCC* — introduced
  `-fstack-usage`/`-fcallgraph-info`
- [avstack](https://github.com/JPHutchins/avstack) — avr stack "worst case usage" tooling
