#include "shared.h"

int second_fn(void)
{
	return 2;
}

struct shared second_instance = {.fn = second_fn, .extra = second_fn};

struct plain plain_second = {.run = second_fn};
