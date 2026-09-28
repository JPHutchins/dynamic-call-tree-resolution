#include "harness.h"

void a(void) {
	observe("a");
}

void b(void) {
	observe("b");
}

[[gnu::noipa]] void (*pick(int const x))(void) {
	return x > 1 ? b : a;
}

[[gnu::noipa]] void run(void ( * const fp)(void)) {
	fp();
}

int main(int argc, [[maybe_unused]] char * argv[argc + 1]) {
	observe("@run");
	run(a);
	run(pick(argc));
	return 0;
}
