
from dataclasses import dataclass

@dataclass
class Value:
    name: str
    ty: object

    def __str__(self):
        return f"{self.name}:{self.ty}"


@dataclass
class Op:
    opcode: str
    result: object
    args: tuple

    def __str__(self):
        lhs = f"{self.result} = " if self.result else ""
        return lhs + self.opcode + " " + " ".join(map(str,self.args))


@dataclass
class FunctionIR:
    name: str
    args: list
    ops: list

    def dump(self):
        out=[f"func @{self.name}"]
        for a in self.args:
            out.append(f"  arg {a}")
        for op in self.ops:
            out.append("  "+str(op))
        return "\n".join(out)
