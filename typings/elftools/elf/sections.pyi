from collections.abc import Iterator
from typing import Literal, TypedDict, overload

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

class SymbolInfo(TypedDict):
	type: Literal[
		"STT_NOTYPE",
		"STT_OBJECT",
		"STT_FUNC",
		"STT_SECTION",
		"STT_FILE",
		"STT_COMMON",
		"STT_TLS",
		"STT_NUM",
		"STT_RELC",
		"STT_SRELC",
		"STT_LOOS",
		"STT_HIOS",
		"STT_LOPROC",
		"STT_HIPROC",
	]
	bind: Literal[
		"STB_LOCAL",
		"STB_GLOBAL",
		"STB_WEAK",
		"STB_NUM",
		"STB_LOOS",
		"STB_HIOS",
		"STB_LOPROC",
		"STB_HIPROC",
	]

class Symbol:
	name: str
	@overload
	def __getitem__(self, name: Literal["st_value", "st_size"]) -> int: ...
	@overload
	def __getitem__(self, name: Literal["st_shndx"]) -> int | str: ...
	@overload
	def __getitem__(self, name: Literal["st_info"]) -> SymbolInfo: ...

class SymbolTableSection(Section):
	def get_symbol(self, n: int) -> Symbol: ...
	def iter_symbols(self) -> Iterator[Symbol]: ...

class ARMAttribute:
	tag: str
	value: int | str

class ARMAttributesSubsubsection:
	def iter_attributes(self) -> Iterator[ARMAttribute]: ...

class ARMAttributesSubsection:
	def iter_subsubsections(self) -> Iterator[ARMAttributesSubsubsection]: ...

class ARMAttributesSection(Section):
	def iter_subsections(self) -> Iterator[ARMAttributesSubsection]: ...
