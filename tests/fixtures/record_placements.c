#include <stdint.h>

struct leaf {
	int value;
	void (*callback)(void);
};

struct empty {};

struct node {
	union {
		struct node * next;
		struct node * head;
	};
	union {
		struct {
			uint8_t prio;
			uint8_t locked;
		};
		uint16_t both;
	};
	int: 3;
	int flags: 5;
	struct leaf leaf;
	struct leaf const pair[2];
	struct leaf grid[2][3];
	struct empty nothing;
};

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

struct node nodes[2];
struct rows rows;
union arms arms;
int counter;
struct leaf * leaves[2];

int main(void) {
	return nodes[0].flags + (int) rows.count + arms.leaf.value + counter + (leaves[0] == nullptr);
}
