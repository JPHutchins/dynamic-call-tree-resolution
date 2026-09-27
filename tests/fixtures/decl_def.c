#include "shared.h"

int target_fn(void)
{
	return 1;
}

struct shared instance = {.fn = target_fn};

struct plain plain_instance = {.run = target_fn};

extern void touch(struct shared *s);

int main(void)
{
	touch(&instance);
	return 0;
}
