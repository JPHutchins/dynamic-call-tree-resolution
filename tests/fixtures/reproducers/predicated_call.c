#include "harness.h"

void a(void) {
	observe("a");
}

void b(void) {
	observe("b");
}

void predicated_call_case(void (*fp)(void), int x);

__asm__(
	".text\n"
	".syntax unified\n"
	".thumb\n"
	".globl predicated_call_case\n"
	".type predicated_call_case, %function\n"
	".thumb_func\n"
	"predicated_call_case:\n"
	"	push {r4, lr}\n"
	"	cmp r1, #1\n"
	"	it gt\n"
	"	blxgt r0\n"
	"	pop {r4, pc}\n"
	".size predicated_call_case, .-predicated_call_case\n"
);

int main(int argc, [[maybe_unused]] char * argv[argc + 1]) {
	observe("@predicated_call_case");
	predicated_call_case(a, 0);
	predicated_call_case(b, argc);
	return 0;
}
