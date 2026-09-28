#include "harness.h"

void a(void) {
	observe("a");
}

void b(void) {
	observe("b");
}

[[gnu::noipa]] void w9(void ( * const fp)(void)) {
	fp();
}

[[gnu::noipa]] void w8(void ( * const fp)(void)) {
	w9(fp);
}

[[gnu::noipa]] void w7(void ( * const fp)(void)) {
	w8(fp);
}

[[gnu::noipa]] void w6(void ( * const fp)(void)) {
	w7(fp);
}

[[gnu::noipa]] void w5(void ( * const fp)(void)) {
	w6(fp);
}

[[gnu::noipa]] void w4(void ( * const fp)(void)) {
	w5(fp);
}

[[gnu::noipa]] void w3(void ( * const fp)(void)) {
	w4(fp);
}

[[gnu::noipa]] void w2(void ( * const fp)(void)) {
	w3(fp);
}

[[gnu::noipa]] void w1(void ( * const fp)(void)) {
	w2(fp);
}

int main(int argc, [[maybe_unused]] char * argv[argc + 1]) {
	observe("@w9");
	w1(b);
	w9(a);
	return 0;
}
