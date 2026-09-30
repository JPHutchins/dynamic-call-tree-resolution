#include "harness.h"

struct operations {
	void ( * const run)(void);
};

struct binding {
	struct operations const * const operations;
};

struct thread_record {
	void ( * const entry)(void const *, void const *, void const *);
	void const * const p1;
	void const * const p2;
	void const * const p3;
};

void run_a(void) {
	observe("run_a");
}

void run_b(void) {
	observe("run_b");
}

static struct operations const operations_a = {.run = run_a};
static struct operations const operations_b = {.run = run_b};
static struct binding const binding_a = {.operations = &operations_a};
[[maybe_unused]] static struct binding const binding_b = {.operations = &operations_b};

[[gnu::noipa]] void worker(
	void const * const bound,
	[[maybe_unused]] void const * const p2,
	[[maybe_unused]] void const * const p3
) {
	struct binding const * const binding = bound;
	binding->operations->run();
}

struct thread_record const records[] = {
	{.entry = worker, .p1 = &binding_a, .p2 = nullptr, .p3 = nullptr},
};

#ifdef ESCAPED
void ( * volatile escaped_entry)(void const *, void const *, void const *) = worker;
#endif

[[gnu::noipa]] void trampoline(struct thread_record const record[static 1]) {
	record->entry(record->p1, record->p2, record->p3);
}

int main(int argc, [[maybe_unused]] char * argv[argc + 1]) {
	observe("@worker");
	trampoline(&records[argc - 3]);
	return 0;
}
