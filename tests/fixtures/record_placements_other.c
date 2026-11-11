#include "record_placements.h"

struct ctx {
	int y[6];
};

struct node other_nodes[1];
struct ctx second_ctx;

int other(void) {
	return other_nodes[0].flags + second_ctx.y[0];
}
