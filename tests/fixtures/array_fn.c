#include <stddef.h>

struct entry {
	void *arg;
	void (*isr)(void);
};

void handler_a(void)
{
}

void handler_b(void)
{
}

struct entry table[2] = {
	{.arg = (void *)1, .isr = handler_a},
	{.arg = (void *)2, .isr = handler_b},
};

void (*cbs[2])(void) = {handler_a, handler_b};

void (*dynamic_cbs[2])(void);

int counts[2] = {0, 0};

__asm__(
	".section .rodata.vector,\"a\"\n"
	".globl vector_table\n"
	".type vector_table, @object\n"
	".size vector_table, 16\n"
	"vector_table:\n"
	"	.quad handler_a\n"
	"	.quad handler_b\n"
);

int main(void)
{
	table[0].isr();
	cbs[1]();
	return dynamic_cbs[0] == NULL;
}
