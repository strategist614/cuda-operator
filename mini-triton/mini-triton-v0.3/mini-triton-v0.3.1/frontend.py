import ast
from ir import Value,Op,FunctionIR

class Frontend:
    def __init__(self):
        self.ops=[]
        self.env={}
        self.args=[]
        self.i=0

    def new(self,t):
        v=Value(f"%{self.i}",t)
        self.i+=1
        return v

    def emit(self,o,t,*a):
        r=self.new(t) if t else None
        self.ops.append(Op(o,r,a))
        return r

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
            return self.emit("tensor_add","tensor<256xf32>",
                             self.expr(n.left),self.expr(n.right))

        if isinstance(n,ast.Call):
            if n.func.id=="arange":
                return self.emit("make_tensor","tensor<256xi32>",256)
            if n.func.id=="load":
                return self.emit("load","tensor<256xf32>",
                                 self.expr(n.args[0]),
                                 self.expr(n.args[1]))
            if n.func.id=="store":
                self.emit("store",None,*[self.expr(x) for x in n.args])
                return None

        raise Exception(ast.dump(n))
