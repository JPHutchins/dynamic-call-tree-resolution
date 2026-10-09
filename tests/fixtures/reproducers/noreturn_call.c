#include "harness.h"

void hook(void) {
	observe("hook");
}

void ( * volatile handler)(void) = hook;

[[noreturn]] [[gnu::noipa]] void halt(void) {
	for (;;) {
	}
}

[[gnu::noipa]] void run(int const count) {
	if (__builtin_expect(count > 5, 1)) {
		halt();
	}
	handler();
}

int main(int argc, [[maybe_unused]] char * argv[argc + 1]) {
	observe("@run");
	run(argc);
	return 0;
}
