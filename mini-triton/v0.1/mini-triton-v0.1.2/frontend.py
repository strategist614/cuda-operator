import ast
from ir import Value, Op, FunctionIR

class Frontend:
    def __init__(self, block=256):
        self.block = block
        self.ops = []
        self.env = {}
        self.id = 0

    def new(self, ty):
        v = Value(f"%{self.id}", ty)
        self.id += 1
        return v

    def emit(self, opcode, ty=None, *args):
        r = self.new(ty) if ty else None
        self.ops.append(Op(opcode, r, args))
        return r

    def compile(self, source):
        tree = ast.parse(source)
        fn = tree.body[0]

        self.env = {
            "X":"X",
            "Y":"Y",
            "Z":"Z",
            "N":"N",
            "BLOCK":self.block
        }

        for stmt in fn.body:
            self.stmt(stmt)

        return FunctionIR(fn.name, self.ops)

    def stmt(self, node):
        if isinstance(node, ast.Assign):
            self.env[node.targets[0].id] = self.expr(node.value)

        elif isinstance(node, ast.Expr):
            self.expr(node.value)

    def expr(self, node):
        if isinstance(node, ast.Name):
            return self.env[node.id]

        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.BinOp):
            a = self.expr(node.left)
            b = self.expr(node.right)

            if isinstance(node.op, ast.Mult):
                return self.emit("mul","i32",a,b)

            if isinstance(node.op, ast.Add):
                return self.emit("add","i32",a,b)

        if isinstance(node, ast.Call):
            name = node.func.id

            if name=="program_id":
                return self.emit("program_id","i32")

            if name=="arange":
                return self.emit("lane_id","i32")

            if name=="load":
                return self.emit("load","f32",*[
                    self.expr(x) for x in node.args
                ])

            if name=="store":
                self.emit("store",None,*[
                    self.expr(x) for x in node.args
                ])
                return None

        if isinstance(node, ast.Compare):
            return self.emit("cmp_lt","pred",
                self.expr(node.left),
                self.expr(node.comparators[0])
            )

        raise Exception(node)
