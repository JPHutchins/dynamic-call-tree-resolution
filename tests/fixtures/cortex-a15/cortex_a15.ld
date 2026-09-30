MEMORY
{
	ram (rwx) : ORIGIN = 0x40000000, LENGTH = 1M
}

ENTRY(start)

SECTIONS
{
	.text : { KEEP(*(.text.start)) *(.text*) *(.rodata*) } > ram
	.data : { *(.data*) } > ram
	.bss : { *(.bss*) *(COMMON) } > ram
	initial_stack = ORIGIN(ram) + LENGTH(ram);
}
