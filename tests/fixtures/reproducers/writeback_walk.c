#include "harness.h"

void f1(void) {
	observe("f1");
}

void f2(void) {
	observe("f2");
}

void f3(void) {
	observe("f3");
}

void ( * const table[])(void) = {f1, f2, f3, nullptr};

[[gnu::noipa]] void walk(void) {
	for (void ( * const * entry)(void) = table; *entry != nullptr; ++entry) {
		(*entry)();
	}
}

int main(int argc, [[maybe_unused]] char * argv[argc + 1]) {
	observe("@walk");
	walk();
	return 0;
}
