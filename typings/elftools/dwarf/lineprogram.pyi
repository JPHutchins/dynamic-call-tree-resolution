from typing import Literal, overload

class FileEntry:
	name: bytes | str
	dir_index: int

class LineProgram:
	@overload
	def __getitem__(self, name: Literal["file_entry"]) -> list[FileEntry]: ...
	@overload
	def __getitem__(self, name: Literal["version"]) -> int: ...
