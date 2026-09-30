struct _static_thread_data {
	void (* init_entry)(void *, void *, void *);
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

void rom_thread(
	[[maybe_unused]] void * const p1,
	[[maybe_unused]] void * const p2,
	[[maybe_unused]] void * const p3
) {}

void ram_thread(
	[[maybe_unused]] void * const p1,
	[[maybe_unused]] void * const p2,
	[[maybe_unused]] void * const p3
) {}

void escaped_thread(
	[[maybe_unused]] void * const p1,
	[[maybe_unused]] void * const p2,
	[[maybe_unused]] void * const p3
) {}

static int rom_binding;

struct _static_thread_data const _k_thread_data_rom_tid = {
	.init_entry = rom_thread,
	.init_p1 = &rom_binding,
};

struct _static_thread_data _k_thread_data_ram_tid = {.init_entry = ram_thread};

struct _static_thread_data const _k_thread_data_null_tid = {.init_entry = nullptr};

struct _static_thread_data const _k_thread_data_escaped_tid = {.init_entry = escaped_thread};

void ( * volatile escaped_entry)(void *, void *, void *) = escaped_thread;

int main(void) {
	return 0;
}
