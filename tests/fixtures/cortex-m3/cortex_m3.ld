MEMORY
{
	flash (rx) : ORIGIN = 0x00000000, LENGTH = 256K
	ram (rwx) : ORIGIN = 0x20000000, LENGTH = 64K
}

ENTRY(reset)

SECTIONS
{
	.text : { KEEP(*(.vectors)) *(.text*) *(.rodata*) } > flash
	.data : { *(.data*) } > ram
	.bss : { *(.bss*) *(COMMON) } > ram
	initial_stack = ORIGIN(ram) + LENGTH(ram);
}
