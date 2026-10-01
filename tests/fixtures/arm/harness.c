#include "harness.h"

#include <stddef.h>

enum semihosting_operation {
	semihosting_write0 = 0x04,
	semihosting_get_cmdline = 0x15,
	semihosting_exit_extended = 0x20,
};

enum semihosting_stop_reason {
	stopped_run_time_error = 0x20023,
	stopped_application_exit = 0x20026,
};

struct semihosting_command_line {
	char * buffer;
	int length;
};

struct semihosting_exit {
	enum semihosting_stop_reason reason;
	int status;
};

enum { command_line_capacity = 128, argument_capacity = 16 };

static inline void semihosting_call(
	enum semihosting_operation const operation,
	void const * const parameters
) {
	register enum semihosting_operation operation_register __asm__("r0") = operation;
	register void const * const parameters_register __asm__("r1") = parameters;
#if __ARM_ARCH_PROFILE == 'M'
	__asm__ volatile ("bkpt 0xab" : "+r"(operation_register) : "r"(parameters_register) : "memory");
#elif defined(__thumb__)
	__asm__ volatile ("svc 0xab" : "+r"(operation_register) : "r"(parameters_register) : "memory");
#else
	__asm__ volatile ("svc 0x123456" : "+r"(operation_register) : "r"(parameters_register) : "memory");
#endif
}

void observe(char const name[static 1]) {
	semihosting_call(semihosting_write0, name);
	semihosting_call(semihosting_write0, "\n");
}

[[noreturn]] static void stop(enum semihosting_stop_reason const reason, int const status) {
	semihosting_call(
		semihosting_exit_extended,
		&(struct semihosting_exit){.reason = reason, .status = status}
	);
	for (;;) {}
}

static int split_arguments(
	char command_line[static 1],
	char * arguments[static argument_capacity]
) {
	int count = 0;
	for (char * cursor = command_line; *cursor != '\0' && count < argument_capacity - 1; ++count) {
		arguments[count] = cursor;
		while (*cursor != '\0' && *cursor != ' ') {
			++cursor;
		}
		while (*cursor == ' ') {
			*cursor++ = '\0';
		}
	}
	arguments[count] = nullptr;
	return count;
}

[[noreturn]] void reset(void) {
	static char command_line[command_line_capacity];
	static char * arguments[argument_capacity];
	semihosting_call(
		semihosting_get_cmdline,
		&(struct semihosting_command_line){.buffer = command_line, .length = command_line_capacity}
	);
	stop(stopped_application_exit, main(split_arguments(command_line, arguments), arguments));
}

extern char initial_stack[];

#if __ARM_ARCH_PROFILE == 'M'
[[noreturn]] static void fault(void) {
	stop(stopped_run_time_error, 1);
}

struct vector_table {
	char * initial_stack;
	void (*reset)(void);
	void (*nmi)(void);
	void (*hard_fault)(void);
};

[[gnu::section(".vectors"), gnu::used]] static struct vector_table const vectors = {
	.initial_stack = initial_stack,
	.reset = reset,
	.nmi = fault,
	.hard_fault = fault,
};
#else
[[gnu::naked, gnu::section(".text.start")]] void start(void) {
	__asm__ volatile ("ldr sp, =initial_stack\n\tb reset");
}
#endif
