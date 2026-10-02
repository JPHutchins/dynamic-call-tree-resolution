from typing import NamedTuple

from elftools.dwarf.compileunit import CompileUnit

class RangeEntry(NamedTuple):
	entry_offset: int
	entry_length: int
	begin_offset: int
	end_offset: int
	is_absolute: bool

class BaseAddressEntry(NamedTuple):
	entry_offset: int
	base_address: int

class RangeLists:
	def get_range_list_at_offset(
		self, offset: int, cu: CompileUnit | None = None
	) -> list[RangeEntry | BaseAddressEntry]: ...
