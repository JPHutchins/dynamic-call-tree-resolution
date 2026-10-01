#include "harness.h"

void initialized_target(void) {
	observe("initialized_target");
}

void (*initialized)(void) = initialized_target;
void (*zeroed)(void);

int main(int argc, char * argv[argc + 1]) {
	for (int index = 0; index < argc; ++index) {
		observe(argv[index]);
	}
	initialized();
	if (argc > 3) {
		zeroed();
	}
	return zeroed == nullptr ? argc : 0;
}
