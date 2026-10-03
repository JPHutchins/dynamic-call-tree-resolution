#include "harness.h"

#include <stdint.h>

unsigned char volatile sink;

void small(void) {
	observe("small");
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

__asm__(
	"\t.section .rodata.callbacks,\"a\"\n"
	"\t.balign 4\n"
	"\t.global callbacks\n"
	"\t.type callbacks, %object\n"
	"\t.size callbacks, 8\n"
	"callbacks:\n"
	"\t.word small - callbacks\n"
	"\t.word deep - callbacks\n"
	"\t.text\n"
);

extern int32_t const callbacks[2];

int volatile choice = 1;

[[gnu::noipa]] void run(int const index) {
	((void (*)(void)) ((char const *) callbacks + callbacks[index]))();
}

int main(int argc, [[maybe_unused]] char * argv[argc + 1]) {
	observe("@main");
	unsigned char volatile frame[1000];
	frame[argc] = 1;
	run(choice);
	return frame[argc] - 1;
}
