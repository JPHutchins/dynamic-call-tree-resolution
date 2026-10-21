# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""salix's replace, typed as its argument where a union defeats salix's own typing (salix#359)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from salix import replace

if TYPE_CHECKING:
	from salix import Struct

	def replace_as_given[S: Struct](instance: S, /, **changes: object) -> S: ...

else:
	replace_as_given = replace
