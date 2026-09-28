#include "harness.h"

void a(void) {
	observe("a");
}

void b(void) {
	observe("b");
}

[[gnu::noipa]] void predicated_store_case(int const x) {
	void ( * volatile slot)(void) = a;
	if (x > 1) {
		slot = b;
	}
	slot();
}

int main(int argc, [[maybe_unused]] char * argv[argc + 1]) {
	observe("@predicated_store_case");
	predicated_store_case(argc);
	return 0;
}
