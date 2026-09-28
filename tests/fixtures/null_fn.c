#include <stddef.h>

void (*null_cb)(void) = NULL;

static __attribute__((unused)) int unused_function(void) {
	return 1;
}

int keep_function(void) {
	return 2;
}

int main(void) {
	if (null_cb != NULL) {
		null_cb();
	}
	return keep_function();
}
