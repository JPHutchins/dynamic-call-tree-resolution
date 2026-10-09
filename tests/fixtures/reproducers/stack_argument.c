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

[[gnu::noipa]] void run(
	[[maybe_unused]] int const w,
	[[maybe_unused]] int const x,
	[[maybe_unused]] int const y,
	[[maybe_unused]] int const z,
	void ( * const fp)(void)
) {
	fp();
}

[[gnu::noipa]] void taken(
	[[maybe_unused]] int const w,
	[[maybe_unused]] int const x,
	[[maybe_unused]] int const y,
	[[maybe_unused]] int const z,
	void ( * const fp)(void)
) {
	fp();
}

void ( * volatile runner)(int, int, int, int, void (*)(void)) = taken;

int main(int argc, [[maybe_unused]] char * argv[argc + 1]) {
	observe("@run");
	run(argc, argc, argc, argc, a);
	run(argc, argc, argc, argc, b);
	observe("@taken");
	runner(argc, argc, argc, argc, c);
	return 0;
}
