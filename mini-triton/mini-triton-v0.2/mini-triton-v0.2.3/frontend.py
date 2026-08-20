import ast
from ir import Value, Op, FunctionIR
from tensor_types import TensorType, PointerType

class Frontend:
    def __init__(self):
        self.ops=[]
        self.args=[]
        self.env={}
        self.id=0

    def new(self,ty):
        v=Value(f"%{self.id}",ty)
        self.id+=1
        return v

    def emit(self,op,ty,*args):
        r=self.new(ty) if ty else None
        self.ops.append(Op(op,r,args))
        return r

    def compile(self,src):
        tree=ast.parse(src)
        fn=[x for x in tree.body if isinstance(x,ast.FunctionDef)][0]

        for a in fn.args.args:
            v=Value(a.arg,PointerType())
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
            a=self.expr(n.left)
            b=self.expr(n.right)
            return self.emit("tensor_add",a.ty,a,b)

        if isinstance(n,ast.Call):
            name=n.func.id

            if name=="arange":
                size=n.args[0].value
                return self.emit("arange",
                    TensorType("i32",(size,)),size)

            if name=="load":
                ptr=self.expr(n.args[0])
                off=self.expr(n.args[1])
                return self.emit("tensor_load",
                    TensorType("f32",off.ty.shape),
                    ptr,off)

            if name=="store":
                self.emit("tensor_store",
                    None,*[self.expr(x) for x in n.args])
                return None

        raise Exception(ast.dump(n))
