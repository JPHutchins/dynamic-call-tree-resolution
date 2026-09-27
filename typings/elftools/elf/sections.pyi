from collections.abc import Iterator
from typing import Literal, overload

class SectionHeader:
	sh_addr: int
	sh_flags: int
	sh_size: int
	sh_type: str

class Section:
	name: str
	header: SectionHeader
	@overload
	def __getitem__(
		self, name: Literal["sh_addr", "sh_entsize", "sh_info", "sh_link", "sh_offset", "sh_size"]
	) -> int: ...
	@overload
	def __getitem__(self, name: Literal["sh_type"]) -> str: ...
	def data(self) -> bytes: ...

class Symbol:
	name: str
	@overload
	def __getitem__(self, name: Literal["st_value", "st_size"]) -> int: ...
	@overload
	def __getitem__(self, name: Literal["st_shndx"]) -> int | str: ...
	@overload
	def __getitem__(self, name: Literal["st_info"]) -> dict[str, str]: ...

class SymbolTableSection(Section):
	def get_symbol(self, n: int) -> Symbol: ...
	def iter_symbols(self) -> Iterator[Symbol]: ...
