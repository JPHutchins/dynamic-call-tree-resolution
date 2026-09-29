#include "harness.h"

struct counter {
	int count;
};

void tick(struct counter * const counter) {
	observe("tick");
	counter->count += 1;
}

void ignore([[maybe_unused]] void * const context) {
	observe("ignore");
}

void ( * handler)(void * context);

[[gnu::noipa]] void (*pick(int const x))(void *) {
	return x > 1 ? (void (*)(void *)) tick : ignore;
}

[[gnu::noipa]] void install(int const x) {
	handler = pick(x);
}

int main(int argc, [[maybe_unused]] char * argv[argc + 1]) {
	install(argc);
	observe("@main");
	handler(&(struct counter){.count = 0});
	return 0;
}
