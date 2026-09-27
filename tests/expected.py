# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""Expected resolution results of the nopie device-model fixture."""

EXPECTED_NOPIE: dict[str, tuple[str, ...]] = {
	"ops_a.open": ("driver_a_open",),
	"ops_a.close": ("driver_a_close",),
	"ops_b.open": ("driver_b_open",),
	"ops_b.close": ("driver_b_close",),
	"dev_a.api.open": ("driver_a_open",),
	"dev_a.api.close": ("driver_a_close",),
	"dev_b.api.open": ("driver_b_open",),
	"dev_b.api.close": ("driver_b_close",),
	"holder.run": ("undef_ptr_target",),
	"node_a.fn": ("node_fn",),
	"plain_cb": ("plain_target",),
	"dev_c.api.open": ("driver_b_open",),
	"dev_c.api.close": ("driver_b_close",),
	"dev_c.context.open": ("driver_a_open",),
	"dev_c.context.close": ("driver_a_close",),
	"dev_a.ops.init": ("dev_init",),
	"dev_b.ops.init": ("dev_init",),
	"dev_c.ops.init": ("dev_init",),
	"holder2.inner.fn": ("anon_fn",),
}

EXPECTED_PATHS: frozenset[str] = frozenset(EXPECTED_NOPIE)
