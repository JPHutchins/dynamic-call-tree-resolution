#include "harness.h"

void a(void) {
	observe("a");
}

void b(void) {
	observe("b");
}

void c(void) {
	observe("c");
}

void d(void) {
	observe("d");
}

void e(void) {
	observe("e");
}

int volatile sink;

[[gnu::noipa]] void switch_case(int const x) {
	void (*fp)(void) = a;
	switch (x) {
	case 0:
		sink = 10;
		fp = b;
		break;
	case 1:
		sink = 11;
		fp = c;
		break;
	case 2:
		sink = 12;
		fp = d;
		break;
	case 3:
		sink = 13;
		fp = e;
		break;
	case 4:
		sink = 14;
		fp = b;
		break;
	case 5:
		sink = 15;
		fp = c;
		break;
	}
	fp();
}

int main(int argc, [[maybe_unused]] char * argv[argc + 1]) {
	observe("@switch_case");
	switch_case(argc);
	return 0;
}
