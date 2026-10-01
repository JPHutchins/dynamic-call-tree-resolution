/*
 * Copyright (c) 2026 JP Hutchins
 * SPDX-License-Identifier: MIT
 */

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

struct sensor_binding {
	struct device const * const device;
	enum sensor_channel const channel;
};

static struct sensor_binding const thermal_binding = {
	.device = DEVICE_DT_GET(DT_NODELABEL(adt7420)),
	.channel = SENSOR_CHAN_AMBIENT_TEMP,
};

static struct sensor_binding const motion_binding = {
	.device = DEVICE_DT_GET(DT_NODELABEL(bmi160)),
	.channel = SENSOR_CHAN_ACCEL_XYZ,
};

static void sensor_thread(
	void * const binding_pointer,
	[[maybe_unused]] void * const unused2,
	[[maybe_unused]] void * const unused3
) {
	struct sensor_binding const * const binding = binding_pointer;
	struct sensor_value readings[3] = {};

	while (true) {
		sensor_sample_fetch(binding->device);
		sensor_channel_get(binding->device, binding->channel, readings);
	}
}

K_THREAD_DEFINE(thermal_tid, 1024, sensor_thread, &thermal_binding, nullptr, nullptr, 5, 0, 0);
K_THREAD_DEFINE(motion_tid, 1024, sensor_thread, &motion_binding, nullptr, nullptr, 5, 0, 0);

static void print_unused_stack(char const name[static 1], k_tid_t const thread) {
	size_t unused = 0;

	if (k_thread_stack_space_get(thread, &unused) == 0) {
		printk("%s unused %zu of %zu\n", name, unused, thread->stack_info.size);
	}
}

int main(void) {
	k_msleep(100);
	print_unused_stack("thermal_tid", thermal_tid);
	print_unused_stack("motion_tid", motion_tid);
	printk("done\n");
	return 0;
}
