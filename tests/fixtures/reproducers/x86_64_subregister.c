#include "harness.h"

void a(void) {
	observe("a");
}

void b(void) {
	observe("b");
}

void (*global)(void) = a;

void subregister_case(int x);

__asm__(
	".text\n"
	".globl subregister_case\n"
	".type subregister_case, @function\n"
	"subregister_case:\n"
	"	mov global(%rip), %rax\n"
	"	cmp $1, %edi\n"
	"	jle 1f\n"
	"	mov $b, %eax\n"
	"1:	jmp *%rax\n"
	".size subregister_case, .-subregister_case\n"
);

int main(int argc, [[maybe_unused]] char * argv[argc + 1]) {
	observe("@subregister_case");
	subregister_case(argc);
	return 0;
}
