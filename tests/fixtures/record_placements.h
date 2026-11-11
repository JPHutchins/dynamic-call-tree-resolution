#pragma once

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
