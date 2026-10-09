#include "harness.h"

void hook(void) {
	observe("hook");
}

struct hook_table {
	unsigned count;
	void ( * const * entries)(void);
};

static void ( * const entries[])(void) = {hook};

struct hook_table const table = {
#ifdef HOOKED
	.count = 1,
#else
	.count = 0,
#endif
	.entries = entries,
};

[[gnu::noipa]] void run(struct hook_table const hooks[static 1]) {
	for (unsigned index = 0; index < hooks->count; ++index) {
		hooks->entries[index]();
	}
}

int main(int argc, [[maybe_unused]] char * argv[argc + 1]) {
	observe("@run");
	run(&table);
	return 0;
}
