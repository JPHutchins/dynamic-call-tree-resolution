from typing import Literal, overload

class FileEntry:
	name: bytes | str
	dir_index: int

class LineProgram:
	@overload
	def __getitem__(self, name: Literal["file_entry"]) -> list[FileEntry]: ...
	@overload
	def __getitem__(self, name: Literal["version"]) -> int: ...
	def get_entries(self) -> list[LineProgramEntry]: ...

class LineState:
	address: int
	file: int
	line: int
	column: int
	end_sequence: bool

class LineProgramEntry:
	state: LineState | None
