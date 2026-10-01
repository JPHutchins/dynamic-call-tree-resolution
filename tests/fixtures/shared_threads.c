struct _static_thread_data {
	void ( * init_entry)(void const *, void const *, void const *);
	void const * init_p1;
	void const * init_p2;
	void const * init_p3;
};

struct operations {
	void ( * const run)(void);
};

void run_small(void) {}

void run_big(void) {}

static struct operations const small_operations = {.run = run_small};
static struct operations const big_operations = {.run = run_big};

void z_thread_entry(
	void ( * const entry)(void const *, void const *, void const *),
	void const * const p1,
	void const * const p2,
	void const * const p3
) {
	entry(p1, p2, p3);
}

void sensor_thread(
	void const * const bound,
	[[maybe_unused]] void const * const p2,
	[[maybe_unused]] void const * const p3
) {
	struct operations const * const operations = bound;
	operations->run();
}

struct _static_thread_data const _k_thread_data_small_tid = {
	.init_entry = sensor_thread,
	.init_p1 = &small_operations,
};

struct _static_thread_data const _k_thread_data_big_tid = {
	.init_entry = sensor_thread,
	.init_p1 = &big_operations,
};

int main(void) {
	return 0;
}
