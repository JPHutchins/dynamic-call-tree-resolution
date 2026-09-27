#include <stddef.h>

void ping(void)
{}

void (*ping_cb)(void) = ping;

extern char undefined_data[] __attribute__((weak));

void * undef_ptr = &undefined_data;

int main(void)
{
	ping_cb();
	return undef_ptr != NULL;
}
