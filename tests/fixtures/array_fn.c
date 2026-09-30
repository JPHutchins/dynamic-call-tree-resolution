#include <stddef.h>

struct entry {
	void * arg;
	void (*isr)(void);
};

static int dummy_arg_a;
static int dummy_arg_b;

void handler_a(void) {}

void handler_b(void) {}

struct entry table[2] = {
	{.arg = &dummy_arg_a, .isr = handler_a},
	{.arg = &dummy_arg_b, .isr = handler_b},
};

void (*cbs[2])(void) = {[0] = handler_a, [1] = handler_b};

void (*dynamic_cbs[2])(void);

int counts[2] = {[0] = 0, [1] = 0};

struct filter {
	void (*rx_cb)(void);
	void * cb_arg;
};

struct bus {
	int id;
	struct filter filters[2];
	void (*hooks[2])(void);
	void (*grid[2][2])(void);
	int levels[2];
};

struct tailed {
	int count;
	void (*tail[])(void);
};

struct bus const rom_bus = {
	.filters = {{.rx_cb = handler_a}, {.rx_cb = handler_b}},
	.hooks = {handler_b, handler_a},
	.grid = {{handler_a, handler_b}, {handler_b, handler_a}},
};

struct bus ram_bus;

struct tailed tailed_bus;

__asm__(
	".section .rodata.vector,\"a\"\n"
	".globl vector_table\n"
	".type vector_table, @object\n"
	".size vector_table, 16\n"
	"vector_table:\n"
	"	.quad handler_a\n"
	"	.quad handler_b\n"
);

int main(void) {
	table[0].isr();
	cbs[1]();
	return dynamic_cbs[0] == NULL;
}
