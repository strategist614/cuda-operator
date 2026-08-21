
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
        r = f"{self.result} = " if self.result else ""
        return r + self.opcode + " " + " ".join(map(str,self.args))


@dataclass
class FunctionIR:
    name: str
    args: list
    ops: list

    def dump(self):
        out=[f"func @{self.name}"]
        out += [f"  arg {a}" for a in self.args]
        out += [f"  {o}" for o in self.ops]
        return "\n".join(out)
