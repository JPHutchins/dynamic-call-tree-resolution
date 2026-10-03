#include "harness.h"

struct ops {
	void (*run)(void);
};

void decoy(void) {
	observe("decoy");
}

void target(void) {
	observe("target");
}

struct ops visible;
struct ops hidden;

[[gnu::noipa]] struct ops const * pick(int const x) {
	return x > 1 ? &hidden : &visible;
}

[[gnu::noipa]] void install(void) {
	visible.run = decoy;
	*(void (**)(void)) (void *) &hidden = target;
}

[[gnu::noipa]] void run(struct ops const * const ops) {
	ops->run();
}

int main(int argc, [[maybe_unused]] char * argv[argc + 1]) {
	install();
	observe("@run");
	run(pick(argc));
	return 0;
}
