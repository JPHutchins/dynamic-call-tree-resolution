#include "harness.h"

unsigned char volatile sink;

void small(void) {
	sink = 1;
}

void deep(void) {
	observe("deep");
	unsigned char volatile buffer[2000];
	for (int index = 0; index < 2000; ++index) {
		buffer[index] = sink;
	}
	sink = buffer[sink];
}

void ( * volatile hook)(void) = small;

[[gnu::noipa]] void (*pick(int const x))(void) {
	return x > 1 ? deep : small;
}

[[gnu::noipa]] void run(void ( * const fp)(void)) {
	fp();
}

int main(int argc, [[maybe_unused]] char * argv[argc + 1]) {
	observe("@main");
	unsigned char volatile frame[1000];
	frame[argc] = 1;
	hook();
#ifdef PARTIAL
	run(small);
	run(pick(argc));
#else
	pick(argc)();
#endif
	return frame[argc] - 1;
}
