#include "harness.h"

void a(void) {
	observe("a");
}

void b(void) {
	observe("b");
}

struct operations {
	void (*function)(void);
	int count;
};

void (*handler)(void);

[[gnu::noipa]] void initialize(struct operations * const operations) {
	operations->function = b;
}

[[gnu::noipa]] void install(void) {
	handler = b;
}

[[gnu::noipa]] void stack_case(void) {
	struct operations operations = {.function = a, .count = 0};
	initialize(&operations);
	operations.function();
}

[[gnu::noipa]] void global_case(void) {
	handler = a;
	install();
	handler();
}

int main(int argc, [[maybe_unused]] char * argv[argc + 1]) {
	observe("@stack_case");
	stack_case();
	observe("@global_case");
	global_case();
	return 0;
}
