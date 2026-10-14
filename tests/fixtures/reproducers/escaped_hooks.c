#include "harness.h"

void a(void) {
	observe("a");
}

void b(void) {
	observe("b");
}

static void (*hidden)(void) = a;
static void (*exposed)(void) = a;

[[gnu::noipa]] void install(void ( * const hook)(void)) {
	hidden = hook;
}

[[gnu::noipa]] void (** slot(void))(void) {
	return &exposed;
}

[[gnu::noipa]] void call_hidden(void) {
	hidden();
}

[[gnu::noipa]] void call_exposed(void) {
	exposed();
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char * argv[argc + 1]) {
	install(b);
	*slot() = b;
	observe("@call_hidden");
	call_hidden();
	observe("@call_exposed");
	call_exposed();
	return 0;
}
