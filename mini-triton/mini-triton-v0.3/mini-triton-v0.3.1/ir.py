from dataclasses import dataclass

@dataclass
class Value:
    name:str
    ty:object
    def __str__(self):
        return f"{self.name}:{self.ty}"

@dataclass
class Op:
    opcode:str
    result:object
    args:tuple
    def __str__(self):
        r=f"{self.result} = " if self.result else ""
        return r+self.opcode+" "+" ".join(map(str,self.args))

@dataclass
class FunctionIR:
    name:str
    args:list
    ops:list

    def dump(self):
        return "\n".join(
            [f"func @{self.name}"]+
            [f"  arg {x}" for x in self.args]+
            [f"  {x}" for x in self.ops]
        )
