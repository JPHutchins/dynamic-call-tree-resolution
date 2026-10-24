/*
 * Copyright (c) 2026 JP Hutchins
 * SPDX-License-Identifier: MIT
 */

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

K_SEM_DEFINE(handoff, 0, 1);

static void take_thread(
	[[maybe_unused]] void * const unused1,
	[[maybe_unused]] void * const unused2,
	[[maybe_unused]] void * const unused3
) {
	while (true) {
		k_sem_take(&handoff, K_FOREVER);
	}
}

static void give_thread(
	[[maybe_unused]] void * const unused1,
	[[maybe_unused]] void * const unused2,
	[[maybe_unused]] void * const unused3
) {
	while (true) {
		k_sem_give(&handoff);
		k_msleep(1);
	}
}

K_THREAD_DEFINE(user_tid, 1024, take_thread, nullptr, nullptr, nullptr, 5, K_USER, 0);
K_THREAD_DEFINE(supervisor_tid, 1024, give_thread, nullptr, nullptr, nullptr, 5, 0, 0);
K_THREAD_ACCESS_GRANT(user_tid, &handoff);

static void print_unused_stack(char const name[static 1], k_tid_t const thread) {
	size_t unused = 0;

	if (k_thread_stack_space_get(thread, &unused) == 0) {
		printk("%s unused %zu of %zu\n", name, unused, thread->stack_info.size);
	}
}

int main(void) {
	k_msleep(100);
	print_unused_stack("user_tid", user_tid);
	print_unused_stack("supervisor_tid", supervisor_tid);
	printk("done\n");
	return 0;
}
