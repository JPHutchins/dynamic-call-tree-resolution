static void target(void) {}

struct ops {
	_Atomic (void (*)(void)) run;
};

struct ops atomic_ops = {.run = target};

int main(void) {
	void (*run)(void) = atomic_ops.run;
	run();
	return 0;
}
