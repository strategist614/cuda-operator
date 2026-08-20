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
        out=[f"func @{self.name}"]
        for op in self.ops:
            lhs=f"{op.result.name} = " if op.result else ""
            out.append(f"  {lhs}{op.opcode} {op.args}")
        return "\n".join(out)
