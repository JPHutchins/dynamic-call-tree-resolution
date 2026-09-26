from collections.abc import Iterator

from elftools.dwarf.compileunit import CompileUnit

class DWARFInfo:
	def iter_CUs(self) -> Iterator[CompileUnit]: ...
