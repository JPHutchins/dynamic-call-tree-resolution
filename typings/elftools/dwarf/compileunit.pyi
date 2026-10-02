from elftools.dwarf.die import DIE

class CompileUnit:
	cu_offset: int
	def get_top_DIE(self) -> DIE: ...
