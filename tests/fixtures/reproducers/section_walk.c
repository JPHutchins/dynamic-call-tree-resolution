#include "harness.h"

struct handler {
	void (*run)(void);
	unsigned priority;
};

void a(void) {
	observe("a");
}

void b(void) {
	observe("b");
}

void c(void) {
	observe("c");
}

[[gnu::section("handlers"), gnu::used]] static struct handler first = {a, 1};
[[gnu::section("handlers"), gnu::used]] static struct handler second = {b, 2};

extern struct handler __start_handlers[];
extern struct handler __stop_handlers[];

[[gnu::noipa]] void walk(void) {
	for (struct handler const * handler = __start_handlers; handler < __stop_handlers; ++handler) {
		handler->run();
	}
}

[[gnu::noipa]] void retarget(void) {
	second.run = c;
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char * argv[argc + 1]) {
	retarget();
	observe("@walk");
	walk();
	return 0;
}
