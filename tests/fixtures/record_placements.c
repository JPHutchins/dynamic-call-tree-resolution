#include "record_placements.h"

typedef struct leaf leaf_row[2];

struct rows {
	leaf_row rows[2];
	unsigned count;
	struct leaf tail[];
};

union arms {
	struct leaf leaf;
	struct node node;
};

struct ctx {
	struct leaf leaf;
	int x[2];
};

struct node nodes[2];
struct rows rows;
union arms arms;
struct ctx first_ctx;
int counter;
struct leaf * leaves[2];

int other(void);

int main(void) {
	return (
		nodes[0].flags +
		(int) rows.count +
		arms.leaf.value +
		first_ctx.x[0] +
		counter +
		(leaves[0] == nullptr) +
		other()
	);
}
