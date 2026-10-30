# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The ``dctr`` application and the options its commands share."""

from typing import TYPE_CHECKING, Final

from cyclopts import App, Parameter

if TYPE_CHECKING:
	from pathlib import Path

app = App(name="dctr")


_MAX_ELF_SIZE: Final = 50 * 1024 * 1024


NARROW_BY_SIGNATURE: Final = Parameter(
	help="narrow a site whose slot the image leaves unset to the functions of the "
	"slot's DWARF signature; unsound, since a cast defeats it"
)


NARROW_BY_FIELD: Final = Parameter(
	help="narrow a site that loads its callee from a struct field to what that field holds: "
	"its initializers in the image and the functions stored into it, from the descriptors.txt "
	"beside the ELF; unsound, since a cast or a memcpy can store a function unseen"
)


ASSUME_NO_RECURSION: Final = Parameter(
	help="assume this function never calls itself, and drop its call to itself; repeat it "
	"for each function. Only a direct self-call is dropped, and the rows that rest on it "
	"say so"
)


ASSUME_UNWRITTEN: Final = Parameter(
	help="assume the program never writes this data object after it loads, so a load from "
	"it reads its value in the image even where a store's address is unknown; repeat it "
	"for each object. Nothing checks it, and the rows that rest on it say so"
)


ASSUME_UPWARD_FRAME_WRITES: Final = Parameter(
	help="assume a frame address the analysis loses track of (passed to a call, stored, or "
	"combined with an unknown value) is written through only at or above itself, so the "
	"frame below it stays known; Arm images only. Nothing checks it, the rows that rest on "
	"it say so, and finding those rows analyzes the image twice"
)


ASSUME_FRAME: Final = Parameter(
	help="assume FUNCTION=BYTES is the frame of a function nothing measures; repeat it for "
	"each function. Nothing checks it, and the rows that rest on it say so"
)


RTOS: Final = Parameter(
	help="the RTOS whose threads start their entries with known arguments: auto detects "
	"it, and none models no RTOS"
)


def oversized(path: Path) -> bool:
	return path.stat().st_size > _MAX_ELF_SIZE
