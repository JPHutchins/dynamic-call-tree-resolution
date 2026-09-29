#include "harness.h"

void a(void) {
	observe("a");
}

void b(void) {
	observe("b");
}

void c(void) {
	observe("c");
}

[[gnu::noipa]] void run(void ( * const fp)(void)) {
	fp();
}

void ( * volatile runner)(void (*)(void)) = run;

[[gnu::noipa]] void wrap(void ( * const fp)(void)) {
	run(fp);
}

int main(int argc, [[maybe_unused]] char * argv[argc + 1]) {
	observe("@run");
	run(a);
	runner(b);
	wrap(c);
	return 0;
}
