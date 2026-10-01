static int target(void) {
	return 1;
}

struct ops {
	int (*run)(void);
	int (*stop)(void);
};

union arm {
	int (*run)(void);
	long value;
};

struct ops const rom_ops = {.run = target};
union arm const rom_arm = {.value = 7};
struct ops ram_ops = {.run = target};
struct ops bss_ops;
int (*written)(void) = target;

int main(void) {
	return rom_ops.run() + ram_ops.run() + written() + (bss_ops.run == nullptr) + rom_arm.value;
}
