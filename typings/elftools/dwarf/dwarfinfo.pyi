from collections.abc import Iterator

from elftools.dwarf.compileunit import CompileUnit
from elftools.dwarf.lineprogram import LineProgram
from elftools.dwarf.ranges import RangeLists

class DWARFInfo:
	def iter_CUs(self) -> Iterator[CompileUnit]: ...
	def range_lists(self) -> RangeLists | None: ...
	def line_program_for_CU(self, cu: CompileUnit) -> LineProgram | None: ...
