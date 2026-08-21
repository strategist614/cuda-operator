
import ast
from ir import Value,Op,FunctionIR


class Frontend:

    def __init__(self):
        self.ops=[]
        self.env={}
        self.args=[]

    def emit(self,op,ty,*args):
        v=Value("%"+str(len(self.ops)),ty)
        self.ops.append(Op(op,v,args))
        return v

    def compile(self,src):
        fn=[x for x in ast.parse(src).body if isinstance(x,ast.FunctionDef)][0]

        for a in fn.args.args:
            v=Value(a.arg,"ptr")
            self.args.append(v)
            self.env[a.arg]=v

        for s in fn.body:
            self.stmt(s)

        return FunctionIR(fn.name,self.args,self.ops)

    def stmt(self,n):
        if isinstance(n,ast.Assign):
            self.env[n.targets[0].id]=self.expr(n.value)
        elif isinstance(n,ast.Expr):
            self.expr(n.value)

    def expr(self,n):
        if isinstance(n,ast.Name):
            return self.env[n.id]

        if isinstance(n,ast.BinOp):
            return self.emit(
                "add",
                "tensor",
                self.expr(n.left),
                self.expr(n.right)
            )

        if isinstance(n,ast.Call):

            if n.func.id=="arange":
                return self.emit(
                    "tensor_value",
                    "tensor<256xf32>",
                    256
                )

            if n.func.id=="load":
                return self.emit(
                    "tensor_load",
                    "tensor<256xf32>",
                    self.expr(n.args[0]),
                    self.expr(n.args[1])
                )

            if n.func.id=="store":
                self.emit(
                    "store",
                    None,
                    *[self.expr(x) for x in n.args]
                )
                return None

        raise Exception(ast.dump(n))
