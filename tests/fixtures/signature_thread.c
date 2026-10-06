struct _static_thread_data {
	void ( * init_entry)(void *, void *, void *);
	void * init_p1;
	void * init_p2;
	void * init_p3;
};

void z_thread_entry(
	void ( * const entry)(void *, void *, void *),
	void * const p1,
	void * const p2,
	void * const p3
) {
	entry(p1, p2, p3);
}

void handle([[maybe_unused]] int const event) {}

void other(void) {}

void ( * const handlers[])(int) = {handle};

void ( * const others[])(void) = {other};

void ( * hook)(int);

int volatile hooked;

void worker(
	[[maybe_unused]] void * const p1,
	[[maybe_unused]] void * const p2,
	[[maybe_unused]] void * const p3
) {
	hook(1);
	hooked = 1;
}

struct _static_thread_data const _k_thread_data_worker_tid = {.init_entry = worker};

int main(void) {
	return 0;
}
