from dataclasses import dataclass

@dataclass
class Value:
    name: str
    ty: str

@dataclass
class Op:
    opcode: str
    result: Value | None
    args: tuple

@dataclass
class FunctionIR:
    name: str
    ops: list

    def dump(self):
        return "\n".join(str(x) for x in self.ops)
