void ping(void) {}

void (*ping_cb)(void) = ping;

[[gnu::weak]] extern char undefined_data[];

void * undef_ptr = &undefined_data;

int main(void) {
	ping_cb();
	return undef_ptr != nullptr;
}
