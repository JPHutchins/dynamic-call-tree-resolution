#include "harness.h"

void small(void) {
	observe("small");
}

void deep(void) {
	observe("deep");
}

[[gnu::noipa]] void run(void ( * const fp)(void)) {
	fp();
}

__asm__(
	"\t.text\n"
	"\t.global install\n"
	"\t.type install, %function\n"
	"install:\n"
	"\tmovw r0, #:lower16:deep\n"
	"\tmovt r0, #:upper16:deep\n"
	"\tb run\n"
);

void install(void);

int main(int argc, [[maybe_unused]] char * argv[argc + 1]) {
	observe("@run");
	run(small);
	install();
	return 0;
}
