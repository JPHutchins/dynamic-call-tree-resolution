from collections.abc import Iterator

CS_ARCH_ARM: int
CS_ARCH_X86: int
CS_MODE_32: int
CS_MODE_64: int
CS_MODE_THUMB: int

class CsError(Exception): ...

class CsMemOperand:
	base: int
	index: int
	scale: int
	disp: int

class CsOperand:
	type: int
	reg: int
	imm: int
	mem: CsMemOperand

class CsShiftOperand:
	type: int
	value: int

class ArmCsOperand(CsOperand):
	shift: CsShiftOperand

class CsInsn:
	id: int
	address: int
	size: int
	bytes: bytes
	mnemonic: str
	op_str: str
	operands: list[CsOperand]

	def regs_access(self) -> tuple[list[int], list[int]]: ...

class Cs:
	detail: bool
	skipdata: bool

	def __init__(self, arch: int, mode: int) -> None: ...
	def disasm(self, code: bytes, offset: int, count: int = 0) -> Iterator[CsInsn]: ...
	def close(self) -> None: ...
