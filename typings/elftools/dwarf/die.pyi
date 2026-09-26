from collections.abc import Iterator
from typing import Any

class AttributeValue:
	name: str
	form: str
	value: Any
	raw_value: int
	offset: int
	indirection_length: int

class DIE:
	tag: str
	offset: int
	attributes: dict[str, AttributeValue]
	has_children: bool
	def get_DIE_from_attribute(self, name: str) -> DIE: ...
	def iter_children(self) -> Iterator[DIE]: ...
