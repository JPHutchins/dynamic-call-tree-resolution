from typing import Literal, overload

class Segment:
	@overload
	def __getitem__(self, name: Literal["p_type"]) -> str: ...
	@overload
	def __getitem__(self, name: Literal["p_vaddr", "p_memsz"]) -> int: ...
