from collections.abc import Iterator

from elftools.dwarf.compileunit import CompileUnit
from elftools.dwarf.ranges import RangeLists

class DWARFInfo:
	def iter_CUs(self) -> Iterator[CompileUnit]: ...
	def range_lists(self) -> RangeLists | None: ...
