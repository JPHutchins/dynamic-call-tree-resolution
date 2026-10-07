#include "harness.h"

unsigned char volatile sink;

void small(void) {
	observe("small");
	sink = 1;
}

[[gnu::used]] void ( * const stored[])(void) = {small};

void deep(void) {
	observe("deep");
	unsigned char volatile buffer[2000];
	for (int index = 0; index < 2000; ++index) {
		buffer[index] = sink;
	}
	sink = buffer[sink];
}

[[gnu::noipa]] void run(void ( * const callback)(void)) {
	callback();
	sink = 2;
}

__asm__(
	"\t.text\n"
	"\t.global install\n"
	"\t.thumb_func\n"
	"\t.type install, %function\n"
	"install:\n"
	"\tmovw r0, #:lower16:deep\n"
	"\tmovt r0, #:upper16:deep\n"
	"\tb run\n"
);

void install(void);

int main(int argc, [[maybe_unused]] char * argv[argc + 1]) {
	observe("@main");
	unsigned char volatile frame[1000];
	frame[argc] = 1;
	install();
	return frame[argc] - 1;
}
