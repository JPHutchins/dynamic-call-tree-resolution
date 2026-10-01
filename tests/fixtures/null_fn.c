void (*null_cb)(void) = nullptr;

[[maybe_unused]] static int unused_function(void) {
	return 1;
}

int keep_function(void) {
	return 2;
}

int main(void) {
	if (null_cb != nullptr) {
		null_cb();
	}
	return keep_function();
}
