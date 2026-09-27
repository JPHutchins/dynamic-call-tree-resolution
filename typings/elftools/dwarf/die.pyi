from collections.abc import Iterator

class AttributeValue:
	name: str
	form: str
	value: int | str | bytes | list[int]
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
