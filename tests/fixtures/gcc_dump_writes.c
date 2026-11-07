#include <stddef.h>

struct dev {
	int count;
	short tag;
	unsigned short flags;
	int regs[4];
	void (*op)(struct dev *);
};

struct node {
	union {
		struct node * next;
		int spare;
	};
	int value;
};

struct pair {
	int a;
	int b;
};

extern void sink(struct dev * d, int * n, int x);
extern int source(void);
extern void take_pair(struct pair * p);
extern void * memcpy(void * restrict dst, void const * restrict src, size_t n);
extern void * memset(void * dst, int c, size_t n);

int counter;
int table[8];
struct dev config;
int word;

void field_store(struct dev * const d, int const n) {
	d->count = n;
}

void variable_store(int const n) {
	counter = n;
}

int static_store(void) {
	static int calls;
	calls = calls + 1;
	return calls;
}

void anonymous_store(struct node * const n, struct node * const next) {
	n->next = next;
}

void record_copy(struct dev * const d, struct dev const * const s) {
	*d = *s;
}

void out_parameter(int * const out, int const v) {
	*out = v;
}

void offset_store(struct dev * const p, unsigned short const v) {
	*(unsigned short *) ((char *) p + 6) = v;
}

void element_stores(struct dev * const d, int const i, int const v) {
	table[i] = v;
	d->regs[i] = v;
}

void pointer_to_pointer(struct dev * * const pp, struct dev * const d) {
	*pp = d;
}

void fills(struct dev * const d, void const * const src, size_t const n) {
	memcpy(&config, src, n);
	memset(d, 0, n);
}

void atomics(struct dev * const d, int const desired) {
	int expected = 0;
	__atomic_fetch_add(&d->count, 1, __ATOMIC_SEQ_CST);
	__atomic_compare_exchange_n(&word, &expected, desired, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
	counter = __atomic_load_n(&word, __ATOMIC_SEQ_CST);
}

void handed(struct dev * const d) {
	sink(d, &counter, 3);
	d->op(d);
}

void result_store(struct dev * const d) {
	d->count = source();
}

struct pair make_pair(int const a) {
	struct pair const made = {.a = a, .b = a + 1};
	return made;
}

void assembly(struct dev * const d, int const v) {
	__asm__ volatile ("str %1, %0" : "=m"(d->count) : "r"(v));
	__asm__ volatile ("nop\n\tnop" : : "r"(d) : "memory");
}

void local_handed(int const a) {
	struct pair local = {.a = a, .b = 2};
	take_pair(&local);
}

extern void sink_int(int * p, int v);

void interior(struct dev * const d, int const v) {
	sink_int((int *) ((char *) d + 8), v);
}
