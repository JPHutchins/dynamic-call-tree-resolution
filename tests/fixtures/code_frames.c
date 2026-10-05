#include "harness.h"

void leaf_push(void);
void calls_leaf(void);
void tail_to_leaf(void);
void unsized(void);
void variable_sub(int bytes);
void indirect(void (*callee)(void));
void self_call(void);

__asm__(
	".syntax unified\n"
	".thumb\n"
	".text\n"
	".global leaf_push\n"
	".thumb_func\n"
	".type leaf_push, %function\n"
	"leaf_push:\n"
	"	push {r4, r5, r6, r7, lr}\n"
	"	pop {r4, r5, r6, r7, pc}\n"
	".size leaf_push, .-leaf_push\n"
	".global calls_leaf\n"
	".thumb_func\n"
	".type calls_leaf, %function\n"
	"calls_leaf:\n"
	"	push {r3, lr}\n"
	"	bl leaf_push\n"
	"	pop {r3, pc}\n"
	".size calls_leaf, .-calls_leaf\n"
	".global tail_to_leaf\n"
	".thumb_func\n"
	".type tail_to_leaf, %function\n"
	"tail_to_leaf:\n"
	"	push {r4, lr}\n"
	"	pop {r4, lr}\n"
	"	b.w leaf_push\n"
	".size tail_to_leaf, .-tail_to_leaf\n"
	".global unsized\n"
	".thumb_func\n"
	".type unsized, %function\n"
	"unsized:\n"
	"	push {r4, r5, lr}\n"
	"	pop {r4, r5, pc}\n"
	".global variable_sub\n"
	".thumb_func\n"
	".type variable_sub, %function\n"
	"variable_sub:\n"
	"	sub sp, sp, r0\n"
	"	add sp, sp, r0\n"
	"	bx lr\n"
	".size variable_sub, .-variable_sub\n"
	".global indirect\n"
	".thumb_func\n"
	".type indirect, %function\n"
	"indirect:\n"
	"	push {r3, lr}\n"
	"	blx r0\n"
	"	pop {r3, pc}\n"
	".size indirect, .-indirect\n"
	".global self_call\n"
	".thumb_func\n"
	".type self_call, %function\n"
	"self_call:\n"
	"	push {r3, lr}\n"
	"	bl self_call\n"
	"	pop {r3, pc}\n"
	".size self_call, .-self_call\n"
);

int main(int argc, [[maybe_unused]] char * argv[argc + 1]) {
	calls_leaf();
	tail_to_leaf();
	unsized();
	variable_sub(argc * 8);
	indirect(leaf_push);
	if (argc > 99) {
		self_call();
	}
	return 0;
}
