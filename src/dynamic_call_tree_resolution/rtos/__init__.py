# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""RTOS adapters: each detects its RTOS in an image and models what it adds."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, assert_never

from dynamic_call_tree_resolution.model import BARE_METAL, RtosModel
from dynamic_call_tree_resolution.rtos import zephyr

if TYPE_CHECKING:
	from dynamic_call_tree_resolution.model import Program


class RtosChoice(StrEnum):
	"""Which RTOS model the analysis uses."""

	AUTO = "auto"
	ZEPHYR = "zephyr"
	NONE = "none"


def rtos_model(
	program: Program, choice: RtosChoice, options: zephyr.Kconfig = zephyr.NO_KCONFIG
) -> RtosModel:
	"""The model the choice selects for the image.

	Raises:
		ValueError: when the chosen RTOS is not detected in the image.
	"""
	match choice:
		case RtosChoice.NONE:
			return BARE_METAL
		case RtosChoice.AUTO | RtosChoice.ZEPHYR:
			match zephyr.detect(program, options):
				case RtosModel() as detected:
					return detected
				case None if choice is RtosChoice.AUTO:
					return BARE_METAL
				case None:
					raise ValueError("no Zephyr thread records or z_thread_entry in the image")
				case _ as unreachable:
					assert_never(unreachable)
		case _ as unreachable:
			assert_never(unreachable)
