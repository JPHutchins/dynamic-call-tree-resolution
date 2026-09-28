#include "harness.h"

void a(void) {
	observe("a");
}

void b(void) {
	observe("b");
}

struct object {
	void (*callback)(void);
};

void (*bss_slot)(void);
void (*data_slot)(void) = a;
struct object objects[2] = {{.callback = a}, {.callback = a}};

[[gnu::noipa]] void (*pick(int const x))(void) {
	return x > 1 ? b : a;
}

[[gnu::noipa]] struct object * lookup(int const index) {
	return &objects[index & 1];
}

[[gnu::noipa]] void write_known(void) {
	bss_slot = a;
}

[[gnu::noipa]] void write_unknown_value(int const x) {
	bss_slot = pick(x);
	data_slot = pick(x);
}

[[gnu::noipa]] void write_unknown_address(int const x) {
	lookup(x)->callback = b;
}

[[gnu::noipa]] void call_bss(void) {
	bss_slot();
}

[[gnu::noipa]] void call_data(void) {
	data_slot();
}

[[gnu::noipa]] void call_object(void) {
	objects[0].callback();
}

int main(int argc, [[maybe_unused]] char * argv[argc + 1]) {
	write_known();
	write_unknown_value(argc);
	write_unknown_address(argc + 1);
	observe("@call_bss");
	call_bss();
	observe("@call_data");
	call_data();
	observe("@call_object");
	call_object();
	return 0;
}
