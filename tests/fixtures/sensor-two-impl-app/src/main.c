/*
 * Copyright (c) 2026 JP Hutchins
 * SPDX-License-Identifier: MIT
 *
 * Two identical threads, each bound to a different existing sensor
 * driver: the ADT7420 temperature sensor and the BMI160 IMU. Both
 * dispatches go through the shared ``sensor`` device class, so the
 * analysis must resolve each thread's ``dev->api`` calls to its own
 * driver to get either thread's stack right.
 */

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/kernel.h>

static struct sensor_value temperature_reading;
static struct sensor_value acceleration_reading[3];

void thermal_thread(void *unused1, void *unused2, void *unused3)
{
	const struct device *thermometer = DEVICE_DT_GET(DT_NODELABEL(adt7420));

	(void)unused1;
	(void)unused2;
	(void)unused3;
	while (1) {
		sensor_sample_fetch(thermometer);
		sensor_channel_get(thermometer, SENSOR_CHAN_AMBIENT_TEMP, &temperature_reading);
	}
}

void motion_thread(void *unused1, void *unused2, void *unused3)
{
	const struct device *imu = DEVICE_DT_GET(DT_NODELABEL(bmi160));

	(void)unused1;
	(void)unused2;
	(void)unused3;
	while (1) {
		sensor_sample_fetch(imu);
		sensor_channel_get(imu, SENSOR_CHAN_ACCEL_XYZ, acceleration_reading);
	}
}

K_THREAD_DEFINE(thermal_tid, 1024, thermal_thread, NULL, NULL, NULL, 5, 0, 0);
K_THREAD_DEFINE(motion_tid, 1024, motion_thread, NULL, NULL, NULL, 5, 0, 0);

int main(void)
{
	return 0;
}
