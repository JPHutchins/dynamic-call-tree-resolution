from collections.abc import Iterator
from typing import Literal

from elftools.elf.sections import Section

class Relocation:
	def __getitem__(
		self, name: Literal["r_offset", "r_info_sym", "r_info_type", "r_addend"]
	) -> int: ...
	def is_RELA(self) -> bool: ...

class RelocationSection(Section):
	def is_RELA(self) -> bool: ...
	def iter_relocations(self) -> Iterator[Relocation]: ...
