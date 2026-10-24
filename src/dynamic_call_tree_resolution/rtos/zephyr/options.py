# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""A Zephyr build's ``.config`` options."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from salix import Struct

if TYPE_CHECKING:
	from collections.abc import Mapping


_KCONFIG_OPTION: Final = re.compile(r"^(?P<option>CONFIG_\w+)=(?P<value>.*)$", re.MULTILINE)
_KCONFIG_INTEGER: Final = re.compile(r"0x[0-9a-fA-F]+|[0-9]+")
_ARCH: Final = "CONFIG_ARCH"
_STACK_NOT_ALL_USABLE: Final = ("CONFIG_ARCH_POSIX", "CONFIG_MPU_STACK_GUARD")
_RANDOM_OFFSET: Final = "CONFIG_STACK_POINTER_RANDOM"
_ALIGN_DOUBLE_WORD: Final = "CONFIG_STACK_ALIGN_DOUBLE_WORD"
_DOUBLE_WORD: Final = 8
_SINGLE_WORD: Final = 4


class Kconfig(Struct):
	"""The options a Zephyr build's ``.config`` sets."""

	options: Mapping[str, str] = {}


NO_KCONFIG: Final = Kconfig()


def kconfig(text: str) -> Kconfig:
	r"""The options a ``.config`` sets, each with its value as written.

	>>> kconfig('CONFIG_IDLE_STACK_SIZE=256\n# CONFIG_FPU is not set\nCONFIG_BOARD="mps2"\n').options
	{'CONFIG_IDLE_STACK_SIZE': '256', 'CONFIG_BOARD': '"mps2"'}
	"""
	return Kconfig(
		options={match["option"]: match["value"] for match in _KCONFIG_OPTION.finditer(text)}
	)


def declared(options: Kconfig, stack_size: int | None) -> int | None:
	r"""A declared stack size, when the build's ``.config`` shows the thread can use all of it.

	Under ``native_sim`` threads run on host stacks, and an MPU stack guard can be carved out of
	the stack.

	>>> declared(kconfig('CONFIG_ARCH="arm"\n'), 1024)
	1024
	>>> declared(kconfig('CONFIG_ARCH="posix"\nCONFIG_ARCH_POSIX=y\n'), 1024) is None
	True
	>>> declared(kconfig('CONFIG_ARCH="arm"\nCONFIG_MPU_STACK_GUARD=y\n'), 1024) is None
	True
	>>> declared(kconfig("CONFIG_SMP=y\n"), 1024) is None
	True
	"""
	return (
		stack_size
		if _ARCH in options.options
		and all(options.options.get(option) != "y" for option in _STACK_NOT_ALL_USABLE)
		else None
	)


def integer(options: Kconfig, name: str) -> int | None:
	r"""An option's value, when it is an integer.

	>>> integer(kconfig("CONFIG_ISR_STACK_SIZE=0x800\nCONFIG_SMP=y\n"), "CONFIG_ISR_STACK_SIZE")
	2048
	>>> integer(kconfig("CONFIG_SMP=y\n"), "CONFIG_SMP") is None
	True
	"""
	match options.options.get(name):
		case str() as value if _KCONFIG_INTEGER.fullmatch(value):
			return int(value, 0)
		case _:
			return None


def random_offset(options: Kconfig) -> int:
	r"""The largest random offset Zephyr can lower a thread's first stack pointer by.

	>>> random_offset(kconfig("CONFIG_STACK_POINTER_RANDOM=64\n"))
	63
	>>> random_offset(NO_KCONFIG)
	0
	"""
	return max((integer(options, _RANDOM_OFFSET) or 0) - 1, 0)


def stack_pointer_alignment(options: Kconfig) -> int:
	r"""``ARCH_STACK_PTR_ALIGN`` on ARM: 8 with ``CONFIG_STACK_ALIGN_DOUBLE_WORD``, else 4.

	Without a ``.config``, the larger.

	>>> stack_pointer_alignment(kconfig('CONFIG_ARCH="arm"\nCONFIG_STACK_ALIGN_DOUBLE_WORD=y\n'))
	8
	>>> stack_pointer_alignment(kconfig('CONFIG_ARCH="arm"\n'))
	4
	>>> stack_pointer_alignment(NO_KCONFIG)
	8
	"""
	return (
		_DOUBLE_WORD
		if _ARCH not in options.options or options.options.get(_ALIGN_DOUBLE_WORD) == "y"
		else _SINGLE_WORD
	)
