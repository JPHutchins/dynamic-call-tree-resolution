#include "harness.h"

struct ops {
	void ( * op)(void);
};

void a(void) {
	observe("a");
}

void b(void) {
	observe("b");
}

[[gnu::noipa]] void nothing(void) {}

[[gnu::noipa]] void run(struct ops const * const ops) {
	ops->op();
}

[[gnu::noipa]] void late(struct ops const * const ops) {
	nothing();
	ops->op();
}

[[gnu::noipa]] void taken(struct ops const * const ops) {
	ops->op();
}

void ( * volatile runner)(struct ops const *) = taken;

[[gnu::noipa]] void run_a(void) {
	struct ops local = {.op = a};
	run(&local);
}

[[gnu::noipa]] void run_b(void) {
	struct ops local = {.op = b};
	run(&local);
}

[[gnu::noipa]] void late_a(void) {
	struct ops local = {.op = a};
	late(&local);
}

[[gnu::noipa]] void taken_b(void) {
	struct ops local = {.op = b};
	runner(&local);
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char * argv[argc + 1]) {
	observe("@run");
	run_a();
	run_b();
	observe("@late");
	late_a();
	observe("@taken");
	taken_b();
	return 0;
}
