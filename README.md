# dynamic-call-tree-resolution

Static resolution of indirect calls in embedded firmware ELF (Executable and Linkable
Format) images, in service of worst-case stack usage analysis.

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
   signature whose address the image takes. With `--narrow-by-field`, a site that loads its callee from a struct field
   narrows to what that field holds: its initializers in the image and the functions the
   build stores into it. Each RTOS static thread is also analyzed on its own, started
   from its record.

Stack depths combine the resulting call graph with GCC's
`-fstack-usage`/`-fcallgraph-info` build artifacts, plus the direct calls the linker kept
and the indirect call sites in the Arm code that `.ci` does not record. Each indirect call also
expands to the fallback: every function whose address the image stores, a non-branch
instruction computes, or an address relocation kept by `--emit-relocs` names. The exceptions
are a site inside a thread's tree that the thread's own analysis resolved, and a call through
a slot the dynamic loader fills with an undefined symbol's address: that call goes to the
symbol, an external function the graph counts as unmeasured. A call whose target can only
be address 0 calls nothing. On an M-profile image, the
fallback also leaves out a handler whose address only the vector table holds: only the
hardware calls it, so it is an entry of its own.

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
z_thread_entry@0x180: bg_thread_main, idle
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
  holds or computes its address: the analysis then starts it with the record's
  `p1`..`p3` instead of unknown arguments. This assumes the kernel's static-thread
  start is the only code that reads the records. `analyze` prints this whole-image
  analysis; `stack --elf` also analyzes each static thread on its own (below).
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
- `z_thread_entry@0x180` is the call that starts every thread. It goes to the entries
  the image's thread creations pass: `bg_thread_main`, the main thread, and `idle`.
  `hello_world` defines no static thread.
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
zephyr.exe                                     EM_386           729 106/144/250      8/7/100
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
resolved slots: 106 | indirect call sites: 83 | not in the image: 286 | membership: names | rtos: zephyr
cmd_prompt_off: unbounded, at least 6264 bytes (recursion: 233, unmeasured: 149)
...
poll_state_thread: unbounded, at least 6056 bytes (recursion: 233, unmeasured: 149)
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
resolved slots: 106 | indirect call sites: 83 | not in the image: 286 | membership: names | rtos: zephyr
poll_state_thread: unbounded, at least 6056 bytes (recursion: 233, unmeasured: 149)
poll_state_thread +80 = 80 bytes (recursion)
bg_thread_main +80 = 160 bytes via indirect: fallback (recursion)
boot_banner +32 = 192 bytes via indirect: fallback (recursion)
...
timer_core_arm +48 = 6056 bytes via static
hwtimer_set_tick_one_shot +0 = 6056 bytes via static (unmeasured)
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
  `__indirect_call` edges of the functions in the image. The `stack --elf` header ends
  with the RTOS model when one is detected.
- `stack --elf` expands every indirect edge to its site candidates plus the fallback,
  every address-taken function, except inside a thread's tree (below). `z_shell_write`
  has no candidates, so its edge is the fallback alone, and every function it reaches
  that way is `indirect: fallback`.
- An `unbounded` entry's number is only a lower bound. The counts name what breaks the
  bound anywhere in the entry's subtree:
  - `recursion`: functions on a cycle;
  - `unmeasured`: functions with neither a `.su` record nor code that can be measured,
    counted as empty frames;
  - `dynamic`: frames GCC could not bound (`alloca` or a VLA);
  - `unresolved`: callers of an indirect call with no candidates.
- A plain `N bytes` bounds every path of the call graph as given. That graph is still
  incomplete ([#96], [#113]).
- `measured`, on a bounded or unbounded row, counts the functions without a `.su` record
  whose frames `stack --elf` measured from their code; the row rests on those
  measurements.
- `narrowed by field` and `narrowed by signature`, on a row, count the callers whose
  indirect call `--narrow-by-field` or `--narrow-by-signature` narrowed. The row rests on
  that flag's assumption, which a cast can break, and `stack --json` names the callers.
- A `binary` edge is a direct call or tail call that the binary records and `.ci` does
  not: one the linker kept (`--emit-relocs`), such as a call to the compiler helper
  `__aeabi_read_tp`, or one decoded from the code of a function measured for its own
  frame, such as `__l_vfprintf`'s to `__ultoa_invert`.
- `stack reservation`, on a thread row, is what the RTOS model adds for the top of the
  thread's stack that the RTOS takes before the thread's entry runs, and the row's
  number includes it.
- `exception frame`, on a thread row, is what the RTOS model adds for the frame an
  interrupt's hardware stacks on the thread's stack, and the row's number includes it.
- `stack`, on a thread row or the main stack's row, is the stack size the build
  declares: a static thread's record, or `CONFIG_MAIN_STACK_SIZE`,
  `CONFIG_IDLE_STACK_SIZE` and `CONFIG_ISR_STACK_SIZE` from the `.config` beside the
  image. `margin` is that size less a bounded row's number, and is negative when the
  number is larger. An unbounded row's `at least` above its stack is not an overflow
  finding: it is the depth of one path the search found, fallback edges included.
- On an M-profile image, each handler only the vector table holds is an entry: the reset
  handler `__start`, exception handlers such as `z_arm_svc`, and the interrupt entry
  `_isr_wrapper`. Its row is that handler's own depth. An assembly handler such as
  `__start` has no `.ci` record, so its row follows the calls the linker kept ([#78]).
- Under Zephyr on an M-profile core, `z_interrupt_stacks` is the row of the main stack:
  the stack exception handlers run on, which the main stack pointer (MSP) addresses
  ([#151]). Its depth is the reset path's, then a chain of nested exceptions, each adding
  its exception frame and its handler's depth:
  - the non-maskable interrupt (NMI) and HardFault, whose priorities are fixed, once each;
  - the deepest of the configurable-priority exceptions: MemManage, BusFault, UsageFault,
    SVCall, DebugMonitor, PendSV, SysTick and each interrupt request (IRQ). A nested
    exception needs a strictly higher priority than the one it interrupts, so the chain
    holds at most one per priority level.
  - The levels come from `arm,num-irq-priority-bits` in the `zephyr.dts` beside the ELF,
    or, without it, from the count of configurable exceptions in the vector table. The
    row names which.
  - Each exception counts once, by its number: `z_arm_hard_fault` handles five of them,
    and `_isr_wrapper` handles every IRQ.
  - `stack --path z_interrupt_stacks` prints the chain, one exception per line, in place
    of a call path.
- Entry points come from the `.ci` graph, which also records functions the linker
  discarded, such as `shell_readline`. With an ELF, calls from functions not in the
  image are ignored, so a function only they call, such as `work_queue_main`, is an
  entry, and entries not in the image are dropped and counted. Without an ELF, every
  `.ci` entry is listed.
- `membership` names what decides "in the image". `linker` reads the final link's map
  (`zephyr_final.map`) and its `--print-gc-sections` listing (`gc-sections.txt`) beside
  the ELF, per object: a function is in the image if its object was linked and its
  section kept. Weak copies and same-named statics are told apart, and the count splits
  into `discarded` (by `--gc-sections`) and `never linked` (an archive member the link
  never pulled in). Without them, `names` matches the names the ELF defines.
- Under `linker`, a kept function's `.ci` call to a libcall GCC records as `<built-in>`
  is dropped when the link did not keep that libcall's code, since a static link
  resolves every call it keeps. GCC records some libcalls that later optimization
  removes, such as `__aeabi_uldivmod` in the BMI160 driver. `summary` lists them as
  `phantom_libcalls` ([#198]).
- With an ELF, each `.ci` node and `.su` record is joined to its function by its
  declaration (file, line, column, within its compilation unit) or by its symbol, so an
  alias such as `z_reschedule_locked` shares its body's frame and calls. Functions that
  share a name are told apart by the shortest end of their unit's path that differs, as
  `uart_stellaris_init@soc_config.c` and `uart_stellaris_init@uart_stellaris.c`;
  `analyze` and `referrers` keep the image's own names.
- With an ELF and an RTOS model, each static thread is an entry named for its thread,
  such as `thermal_tid`. It is `z_thread_entry`'s frame and calls, with the indirect call
  that starts the thread going to that thread's entry alone. The entry function is then
  no longer an entry of its own.
- The threads Zephyr creates for itself while it starts are entries the same way, named
  for their thread objects: `z_main_thread` runs `bg_thread_main`, and `z_idle_threads`
  runs `idle`. Their trees below the entry are the image's.
- Below that, a thread's tree comes from an analysis of the thread alone, started from
  its record. A site that analysis tracked to known functions expands to those functions
  without the fallback, provided it decoded as many indirect calls in the function as
  `.ci` records. So two threads that share an entry but pass it different bindings get
  different trees. Any other site expands as it does everywhere else, and below it the
  thread's tree is the image's.
- `stack --json` carries the full function names behind each count.
- `stack --path ENTRY` prints that entry alone, then its deepest path, one function per
  line: its frame, the running total, the edge it is called through (`static`,
  `indirect: candidate`, `indirect: field`, `indirect: signature`, `indirect: fallback`,
  `thread record`, `system thread`, or `binary`), and what its own frame, calls, or
  cycle add to an unbounded depth. The last total is the entry's depth.

## Per-thread trees

The sensor-threads fixture (`tests/fixtures/sensor-threads-app`, built at
`$DCTR_FIXTURES/sensor-threads`) runs one entry function, `sensor_thread`, as two Zephyr
static threads. Each `K_THREAD_DEFINE` passes it a `static const` binding as `p1`:
`thermal_tid` an ADT7420 temperature sensor on an emulated I2C bus, and `motion_tid` a
BMI160 accelerometer on an emulated SPI bus.

The whole-image analysis sees one `sensor_thread`, so its dispatch is the union of both
bindings, and the drivers' own bus calls stay unresolved:

```console
$ dctr analyze $DCTR_FIXTURES/sensor-threads/zephyr/zephyr.elf
rtos: zephyr (detected: z_thread_entry, struct _static_thread_data)
thread motion_tid: sensor_thread (seeded from its record)
thread thermal_tid: sensor_thread (seeded from its record)
...
z_thread_entry@0xc44: bg_thread_main, sensor_thread, idle
...
sensor_thread@0x2fe2: adt7420_sample_fetch, bmi160_sample_fetch
sensor_thread@0x2fee: adt7420_channel_get, bmi160_channel_get
...
i2c_write_read.constprop.0@0x3308: <unresolved>
...
```

`referrers` lists each place that holds or computes an address-taken function's
address. A thread starts from its record's arguments only when its records are its
entry's only referrers, as they are for `sensor_thread`:

```console
$ dctr referrers $DCTR_FIXTURES/sensor-threads/zephyr/zephyr.elf
...
bg_thread_main: z_cstart@0x2118
...
idle: z_init_cpu@0x203c
...
sensor_thread: _k_thread_data_motion_tid@0x413c, _k_thread_data_thermal_tid@0x416c
...
z_thread_entry: arch_new_thread@0x124c, arch_switch_to_main_thread@0x127c
...
```

With `--narrow-by-field`, each bus emulator's call narrows to the one emulator on its
bus ([#159]):

```console
$ dctr analyze $DCTR_FIXTURES/sensor-threads/zephyr/zephyr.elf --narrow-by-field
...
i2c_emul_transfer@0x325a: adt7420_emul_transfer_i2c (narrowed by field struct i2c_emul_api.transfer, unsound under casts)
...
spi_emul_io@0x3b84: bmi160_emul_io_spi (narrowed by field struct spi_emul_api.io, unsound under casts)
...
```

`stack` analyzes each thread on its own, from its record:

```console
$ dctr stack $DCTR_FIXTURES/sensor-threads --elf $DCTR_FIXTURES/sensor-threads/zephyr/zephyr.elf
resolved slots: 127 | indirect call sites: 26 | not in the image: 337 (discarded: 215, never linked: 122) | membership: linker | rtos: zephyr
z_interrupt_stacks: unbounded, at least 10592 bytes (recursion: 50, measured: 13, nested exceptions: 10, priority levels: 8 (devicetree), exception frame: 36 bytes each, stack: 2048 bytes)
motion_tid: unbounded, at least 1168 bytes (recursion: 50, measured: 8, stack reservation: 16 bytes, exception frame: 36 bytes, stack: 1024 bytes)
thermal_tid: unbounded, at least 1156 bytes (recursion: 50, measured: 8, stack reservation: 16 bytes, exception frame: 36 bytes, stack: 1024 bytes)
...
```

The main stack's row stacks on the reset path the exceptions that can nest there, each
by its exception number:

```console
$ dctr stack $DCTR_FIXTURES/sensor-threads --elf $DCTR_FIXTURES/sensor-threads/zephyr/zephyr.elf --path z_interrupt_stacks
resolved slots: 127 | indirect call sites: 26 | not in the image: 337 (discarded: 215, never linked: 122) | membership: linker | rtos: zephyr
z_interrupt_stacks: unbounded, at least 10592 bytes (recursion: 50, measured: 13, nested exceptions: 10, priority levels: 8 (devicetree), exception frame: 36 bytes each, stack: 2048 bytes)
(exception 1) __start +1016 = 1016 bytes
(exception 4) z_arm_hard_fault +36 +1048 = 2100 bytes
(exception 5) z_arm_hard_fault +36 +1048 = 3184 bytes
(exception 6) z_arm_hard_fault +36 +1048 = 4268 bytes
(exception 12) z_arm_hard_fault +36 +1048 = 5352 bytes
(exception 15) sys_clock_isr +36 +1040 = 6428 bytes
(exception 16) _isr_wrapper +36 +976 = 7440 bytes
(exception 17) _isr_wrapper +36 +976 = 8452 bytes
(exception 18) _isr_wrapper +36 +976 = 9464 bytes
(exception 3) z_arm_hard_fault +36 +1048 = 10548 bytes
(exception 2) z_arm_nmi +36 +8 = 10592 bytes
```

```console
$ dctr stack $DCTR_FIXTURES/sensor-threads --elf $DCTR_FIXTURES/sensor-threads/zephyr/zephyr.elf --path thermal_tid
resolved slots: 127 | indirect call sites: 26 | not in the image: 337 (discarded: 215, never linked: 122) | membership: linker | rtos: zephyr
thermal_tid: unbounded, at least 1156 bytes (recursion: 50, measured: 8, stack reservation: 16 bytes, exception frame: 36 bytes, stack: 1024 bytes)
thermal_tid +8 = 8 bytes
sensor_thread +32 = 40 bytes via thread record
adt7420_sample_fetch +32 = 72 bytes via indirect: candidate
i2c_write_read +32 = 104 bytes via static
i2c_emul_transfer +32 = 136 bytes via indirect: candidate
adt7420_init +8 = 144 bytes via indirect: fallback (recursion)
...
```

```console
$ dctr stack $DCTR_FIXTURES/sensor-threads --elf $DCTR_FIXTURES/sensor-threads/zephyr/zephyr.elf --path motion_tid
resolved slots: 127 | indirect call sites: 26 | not in the image: 337 (discarded: 215, never linked: 122) | membership: linker | rtos: zephyr
motion_tid: unbounded, at least 1168 bytes (recursion: 50, measured: 8, stack reservation: 16 bytes, exception frame: 36 bytes, stack: 1024 bytes)
motion_tid +8 = 8 bytes
sensor_thread +32 = 40 bytes via thread record
bmi160_sample_fetch +24 = 64 bytes via indirect: candidate
bmi160_byte_read +0 = 64 bytes via static
bmi160_read +4 = 68 bytes via static
bmi160_read_spi +48 = 116 bytes via indirect: candidate
spi_emul_io +32 = 148 bytes via indirect: candidate
adt7420_init +8 = 156 bytes via indirect: fallback (recursion)
...
```

- Down to its bus emulator, each thread calls only its own driver: `thermal_tid` the
  ADT7420 and `motion_tid` the BMI160. `i2c_write_read`, which `analyze` leaves
  unresolved, calls `i2c_emul_transfer` alone in `thermal_tid`'s tree.
- The trees rejoin below the bus emulators. `i2c_emul_transfer` and `spi_emul_io` find
  their target in a list that init code builds in RAM, so their calls expand to the
  fallback. From there both threads share the image's tree and its recursion ([#151],
  [#159]).
- Both rows are unbounded. Each `at least` is one depth-first path through that shared
  cycle, and it differs by where the thread enters it.
- On QEMU, the fixture prints each thread's unused stack (`CONFIG_INIT_STACKS`,
  `k_thread_stack_space_get`), and `test_sensor_threads` pins the output:

  ```text
  thermal_tid unused 840 of 1024
  motion_tid unused 864 of 1024
  ```

- Checking each measured mark against a static bound needs bounded rows, and is
  deferred to [#169]. With `--narrow-by-field`, `motion_tid` is bounded, and its mark is
  checked (below).

With `--narrow-by-field`, each thread's tree stays below its own bus emulator:

```console
$ dctr stack $DCTR_FIXTURES/sensor-threads --elf $DCTR_FIXTURES/sensor-threads/zephyr/zephyr.elf --narrow-by-field
resolved slots: 127 | indirect call sites: 26 | not in the image: 337 (discarded: 215, never linked: 122) | membership: linker | narrowed by field | rtos: zephyr
...
thermal_tid: unbounded, at least 236 bytes (recursion: 1, measured: 3, narrowed by field: 1, stack reservation: 16 bytes, exception frame: 36 bytes, stack: 1024 bytes)
...
motion_tid: 236 bytes (measured: 3, narrowed by field: 1, stack reservation: 16 bytes, exception frame: 36 bytes, stack: 1024 bytes, margin: 788 bytes)
...
```

- `motion_tid` is bounded. Three of its frames are measured from code: `memset`,
  `__aeabi_ldivmod` and `__aeabi_read_tp`, prebuilt library functions without `.su`
  records. `.ci` does not record the call to `__aeabi_read_tp`, a `binary` edge. Its
  bound includes the stack reservation and the exception frame, and its QEMU mark above
  is within it; `test_sensor_threads` asserts that, and how far below the bound the mark
  is ([#169]).
- `thermal_tid` recurses through `i2c_emul_transfer`, which can forward a transfer to
  another bus. That recursion is in the code, and no narrowing removes it.

In this build the forward list is empty, so `i2c_emul_transfer` never calls itself.
Stating that bounds `thermal_tid` too:

```console
$ dctr stack $DCTR_FIXTURES/sensor-threads --elf $DCTR_FIXTURES/sensor-threads/zephyr/zephyr.elf --narrow-by-field --assume-no-recursion i2c_emul_transfer
resolved slots: 127 | indirect call sites: 26 | not in the image: 337 (discarded: 215, never linked: 122) | membership: linker | narrowed by field | assumed no recursion: i2c_emul_transfer | rtos: zephyr
...
motion_tid: 236 bytes (measured: 3, narrowed by field: 1, stack reservation: 16 bytes, exception frame: 36 bytes, stack: 1024 bytes, margin: 788 bytes)
thermal_tid: 236 bytes (measured: 3, assumed no recursion: 1, narrowed by field: 1, stack reservation: 16 bytes, exception frame: 36 bytes, stack: 1024 bytes, margin: 788 bytes)
...
```

- Both QEMU marks above are within these bounds, and `test_sensor_threads` asserts it,
  and how far below each bound each mark is ([#169]). Each row's `stack` is the size QEMU
  prints, and its `margin` is no more than what QEMU leaves unused.
- Each row starts at the top of its stack, as QEMU's mark does. Zephyr gives that top
  to thread-local storage (TLS) before the thread runs, so the stack pointer reaches
  `z_thread_entry` one stack reservation below it. `test_sensor_threads` reads that
  stack pointer with gdb on QEMU, for both threads and main, and `test_rtos` for main
  and idle in `hello_world` ([#151]).
- `z_idle_threads` is bounded, but its deepest path is `z_thread_entry`'s own call to
  abort a thread whose entry returns, which `idle` never does, so its small margin
  understates the room `idle` has:

```console
$ dctr stack $DCTR_FIXTURES/sensor-threads --elf $DCTR_FIXTURES/sensor-threads/zephyr/zephyr.elf --path z_idle_threads
resolved slots: 127 | indirect call sites: 26 | not in the image: 337 (discarded: 215, never linked: 122) | membership: linker | rtos: zephyr
z_idle_threads: 236 bytes (measured: 1, stack reservation: 16 bytes, exception frame: 36 bytes, stack: 256 bytes, margin: 20 bytes)
z_idle_threads +8 = 8 bytes
z_impl_k_thread_abort +0 = 8 bytes via static
...
(stack reservation) +16 = 200 bytes
(exception frame) +36 = 236 bytes
```

Zephyr's own `samples/synchronization`, built at `$DCTR_FIXTURES/synchronization`,
starts `thread_b` from a record and `thread_a` with `k_thread_create`. The call that
starts every thread goes to both:

```console
$ dctr analyze $DCTR_FIXTURES/synchronization/zephyr/zephyr.elf
rtos: zephyr (detected: z_thread_entry, struct _static_thread_data)
thread thread_b: thread_b_entry_point (seeded from its record)
...
z_thread_entry@0x244: thread_a_entry_point, thread_b_entry_point, bg_thread_main, idle
...
```

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
- A link with `--emit-relocs` loads as the same link without it, plus the references
  it keeps: the linker already applied them, so they are not applied again.
- `EM_ARM`, `EM_386`, and `EM_X86_64`; other machines are rejected. `EM_ARM` code is
  decoded as A32 inside the spans of `$a` mapping symbols and as Thumb elsewhere, with
  the M-profile encodings when the build attributes name an M-profile core.
- Zephyr is the only RTOS modeled, and only its static threads get trees of their own:
  a thread created with `k_thread_create` starts its entry with unknown arguments.
- The call in `z_thread_entry` that starts every thread goes to the entries the image's
  thread creations pass. This assumes that the kernel's static-thread start passes
  exactly its records' entries ([#185]), and holds only while `arch_new_thread` and
  `arch_switch_to_main_thread` alone hold `z_thread_entry`'s address. Otherwise, or
  when a creation's entry is unknown, the call is unresolved.

### Call targets

- On the four Arm fixtures, `test_qemu_targets` runs each image on QEMU under gdb and
  checks every indirect call it makes against what the analysis reports: the site's
  candidates, or the fallback where it has none; what `--narrow-by-field` narrows the
  site to; and, on a static thread, what the thread's own analysis resolved ([#75]). A
  run covers only the calls its code reaches.
- On the same fixtures, `test_decode` checks the decoding against GNU objdump: every
  instruction starts where objdump starts one, every direct branch and call goes where
  objdump says, objdump's register branches are exactly the indirect call sites, and
  every other write to `pc` is a return or a jump table the decoder follows ([#75]).
- On every Zephyr fixture, `test_invariants` checks that every candidate `analyze`
  reports is a callee of its site in the stack graph; that every target a resolution or
  a narrowing names, globally or in a thread's own analysis, is in the fallback, so
  removing a resolution never lowers a bound; and that no entry is deeper with its
  candidates alone than with the fallback too ([#75]). On counter-su (x86), the first
  fails ([#228]).
- A read-only slot's value is its value in the image as linked. A writable slot's
  candidates are its initializer and the values stored to it, so a store whose address
  or value is unknown leaves it unresolved.
- On a dynamically linked image, a slot that a dynamic relocation fills with an undefined
  symbol reads as unknown, not as the zero the image holds. A call that reads its target
  from such a slot, such as x86 `call *__libc_start_main@GOTPCREL(%rip)` in glibc's
  `_start`, goes to that symbol as an external function, and so does a call through a
  register loaded from such a slot, such as the `__gmon_start__` call in x86-64 `_init`.
  On i386, `_init` finds the slot through `__x86.get_pc_thunk.bx`, which the analysis
  doesn't follow, so that call stays unresolved and expands to the fallback ([#228]).
- A site whose target can only be address 0 calls nothing: no function starts there,
  and calling it faults. `analyze` prints `<null>` for it, and it adds no edge. This
  holds when the code sets the 0 (glibc's `register_tm_clones` loads a weak undefined
  symbol the linker resolved to 0) or a read-only slot holds it. A 0 read from writable
  memory still expands to the fallback, since a store the analysis misses could change
  it ([#228]).
- *Exact* means one candidate in the image as linked, not the only function the site
  can call at runtime.
- A store whose address the analysis cannot compute makes every writable address
  unknown, so a site that reads its target from RAM is unresolved unless its own path
  wrote that target.
- Value-set analysis per-site sets are refinements, not over-approximations: its
  control-flow graph misses x86 `notrack` switches, and it does not model a few kinds of
  x86 write ([#113]). An x86 register write reaches its whole family: a 32-bit write on
  x86-64 zero-extends into its 64-bit register, and a 16- or 8-bit write leaves the
  family's value unknown. A read of a narrower register than the family's widest is
  unknown.
- The value-set analysis reads a function symbol of size zero (hand-written assembly) up
  to the next function, so the calls and stores made there count ([#219]). Code under
  no function symbol is not read ([#96]).
- Inside a thread's tree, a site the thread's analysis resolved drops the fallback, so a
  gap in that analysis drops a target there. With `--narrow-by-field`, so does a site
  its field narrowed, and the edge reads `indirect: field`; with `--narrow-by-signature`,
  so does a site its slot's signature narrowed ([#197]).
- Zephyr's I2C and SPI emulators find their target in a list that init code builds in
  RAM, so their sites stay unresolved, and two threads' trees rejoin below them ([#151],
  [#159]). With `--narrow-by-field`, each narrows to the emulator on its own bus.
- `--narrow-by-signature` compares DWARF signatures for equality, so a cast defeats
  it. It is off by default. It narrows a site whose target comes from one slot the
  analysis can name: read by the call itself, as in x86 `call *(%rax)`, or loaded into the
  register the call branches through, as in Arm `ldr r3, [r3]; blx r3` ([#211]).
- `--assume-no-recursion FUNCTION`, repeated for each function, states that the function
  never calls itself and drops its call to itself. Only a direct self-call is dropped: a
  function on a longer cycle stays recursive. Nothing checks the assumption; the header
  and every row that rests on it name it ([#169], [#202]).
- `--narrow-by-field` assumes that every function stored into a field is stored as that
  field. A cast, a `memcpy` or a union member can store one it never sees
  (`tests/fixtures/reproducers/field_cast.c`), so it is off by default. It reads the
  `descriptors.txt` beside the ELF; the plugin's pass runs only with the optimizer, so an
  `-O0` build records nothing to narrow by. A field that something other than a function
  is stored into, such as a parameter, does not narrow.
- The fallback holds every function address the image stores, that one instruction or
  a `movw`/`movt` pair in a function symbol computes, or that an address relocation kept
  by `--emit-relocs` names, including a relative offset such as `.word f - table`. A
  function symbol of size zero (hand-written assembly) is read up to the next function.
  Without `--emit-relocs`, a function pointer built by other arithmetic, or in code
  outside every function symbol, is missed ([#96]).
- A handler leaves the fallback only when every reference to its address is a slot of
  the vector table. That assumes no software loads a handler from the table and calls
  it; such a load, when the value-set analysis tracks it, still resolves to the handler.
  The table is the one that holds the ELF entry point after a nonzero, 8-byte-aligned
  initial stack pointer.
- Code without `.ci` records (assembly, prebuilt libraries) joins the stack call graph
  only through the direct calls the linker kept, and `native_sim` host code is absent
  ([#78], [#163]).

### Stack depths

- A depth is the deepest path the search found over the call graph; it is a bound
  only if that graph is complete, and today it is not (above).
- In an entry that reaches a large cycle, `at least` is the depth of one path through
  the cycle that a depth-first walk finds, not the deepest. More edges, such as the
  fallback's, can print a smaller number.
- Without an ELF, entry points come from `.ci`, including functions the linker
  discarded, and a function only they call is not an entry ([#78]).
- Under Zephyr on an M-profile core, a thread row adds the frame the hardware stacks
  on the thread's stack when an interrupt arrives: 8 words without an FPU, 26 with one,
  plus a word that may align the stack. It is added once, since handlers run on the
  main stack, whose row is `z_interrupt_stacks`. Zephyr's context switch saves registers
  in the thread object, not on its stack. Other cores' exception stacking is not
  modeled.
- The main stack's row reads no priorities. It assumes any configurable-priority
  exception can nest at any level, so it over-counts where priorities, priority grouping
  or masking keep exceptions from nesting. Reading the image's priorities is a later,
  opt-in narrowing ([#151]). Its base, the reset path, is `__start`'s deepest path,
  fallback edges included.
- Under Zephyr on Arm, a thread row also adds the top of the stack Zephyr takes before
  the thread runs:
  - with thread-local storage, the image's TLS sections, each rounded up to its
    alignment as Zephyr counts them, and two toolchain pointers. The
    `.config` beside the image says whether the build has it
    (`CONFIG_THREAD_LOCAL_STORAGE`), and so does the `arch_tls_stack_setup` symbol, when
    link-time optimization has not inlined it;
  - with `CONFIG_STACK_POINTER_RANDOM`, the largest random offset;
  - rounded up to an 8-byte boundary, or a 4-byte one when the `.config` does not set
    `CONFIG_STACK_ALIGN_DOUBLE_WORD`.

  Zephyr can take more, which the model leaves out: userspace-local data, and stack
  canaries placed with TLS. x86 images get neither this nor the exception frame
  ([#151]).
- A row's `stack` is the size the build declares. Zephyr's usable stack is that size
  rounded up to the stack alignment, so it is at least the declared size, and a `margin`
  can understate the room but not overstate it. A row shows its `stack` only when the
  `.config` beside the image is read and says the thread can use all of it: not under
  `native_sim`, whose threads run on host stacks, and not with `CONFIG_MPU_STACK_GUARD`,
  which can carve its guard out of the stack ([#151]).
- With `--elf`, a function the graph reaches without a `.su` record, such as prebuilt
  library code, gets a frame measured from its Arm code: the deepest its stack pointer
  goes, or the depth at a call plus that callee's measured depth. When that whole depth
  can't be measured, because the function calls through a register or a callee can't
  be measured, it gets its own frame from its code instead, and its calls count through
  the graph: direct calls as `binary` edges, and calls through a register as indirect
  sites, resolved to candidates or expanded to the fallback ([#214]). A function whose
  stack pointer becomes unknown, or that branches through a register other than to
  return, stays unmeasured. A function symbol of size zero (hand-written assembly) is
  measured up to the next function. x86 code is not measured: its `.su` records appear to count
  the return address, and that convention is unchecked ([#195]).
- With `--elf`, a direct call the linker kept joins the graph when `.ci` lacks it and its
  caller has a `.su` frame or code that cannot be measured. A measured frame already
  counts its callees, so its calls are not added again. A tail call counts as a call,
  which can over-count. Without `--emit-relocs`, the image keeps no call relocations, and
  only the calls decoded from a function measured for its own frame join. A call inside
  a function symbol of size zero belongs to that function, up to the next function, as
  its measured frame does ([#78]).
- With `--elf`, an indirect call site found in a function's Arm code joins the graph when
  `.ci` records fewer indirect calls for that function. That covers a call from inline
  assembly, such as Zephyr's `arch_switch_to_main_thread`, and a call from code built
  without `-fcallgraph-info` whose frame can't be measured ([#75]). x86 sites are not
  added ([#228]).

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
a machine builds it. Each testbed ships its whole build folder: the ELF, `.config`, the
generated devicetree and headers, the link maps, and the `.su` and `.ci` files where
they were built. Nix store paths in it are scrubbed, so it refers to nothing outside
itself. The `qemu_cortex_m3` testbeds link with `--emit-relocs` and
`--print-gc-sections`, and each also ships the final link's `gc-sections.txt` next to
the ELF, taken from the build log ([#150]).

Each Arm testbed also ships `descriptors.txt`: for every indirect call, what GCC saw it
load its callee from (a struct field, a parameter, a variable or an array element), and
every store of a function pointer into a field or a variable. A read-only GCC plugin in
`testbeds/plugin` records them. The Zephyr SDK compiler cannot load plugins, so
`replay.py` compiles each C file again with Arm GNU, twice, and fails unless the code
with the plugin disassembles the same as without it ([#158]). For a local build:

```sh
nix develop
uv run camas descriptors --NAME=hello   # after `camas testbeds`, into .camas/build/hello
```

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
[#75]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/75
[#78]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/78
[#96]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/96
[#113]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/113
[#131]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/131
[#150]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/150
[#151]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/151
[#158]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/158
[#159]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/159
[#163]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/163
[#169]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/169
[#185]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/185
[#195]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/195
[#197]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/197
[#198]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/198
[#202]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/202
[#211]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/211
[#214]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/214
[#219]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/219
[#228]: https://github.com/JPHutchins/dynamic-call-tree-resolution/issues/228
