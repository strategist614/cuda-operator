import ast

from ir import Value, Op, FunctionIR


class Frontend:

    def __init__(self):
        self.ops = []
        self.env = {}
        self.idx = 0


    def new(self, ty):

        v = Value(
            f"%{self.idx}",
            ty
        )

        self.idx += 1

        return v


    def emit(self, op, ty=None, *args):

        result = None

        if ty:
            result = self.new(ty)

        self.ops.append(
            Op(
                op,
                result,
                args
            )
        )

        return result


    def compile(self, src):

        tree = ast.parse(src)


        # 找真正的 kernel function

        fn = None

        for node in tree.body:

            if isinstance(
                node,
                ast.FunctionDef
            ):
                fn = node
                break


        if fn is None:
            raise Exception(
                "kernel function not found"
            )


        for stmt in fn.body:
            self.stmt(stmt)


        return FunctionIR(
            fn.name,
            self.ops
        )


    def stmt(self, node):

        if isinstance(
            node,
            ast.Assign
        ):

            name = node.targets[0].id

            value = self.expr(
                node.value
            )

            self.env[name] = value


        elif isinstance(
            node,
            ast.Expr
        ):

            self.expr(
                node.value
            )


    def expr(self, node):


        # x
        if isinstance(
            node,
            ast.Name
        ):

            return self.env.get(
                node.id,
                node.id
            )


        # 123
        if isinstance(
            node,
            ast.Constant
        ):

            return node.value



        # a+b / a*b

        if isinstance(
            node,
            ast.BinOp
        ):

            a = self.expr(
                node.left
            )

            b = self.expr(
                node.right
            )


            if isinstance(
                node.op,
                ast.Add
            ):

                return self.emit(
                    "add",
                    "i32",
                    a,
                    b
                )


            if isinstance(
                node.op,
                ast.Mult
            ):

                return self.emit(
                    "mul",
                    "i32",
                    a,
                    b
                )



        # function call

        if isinstance(
            node,
            ast.Call
        ):


            if not isinstance(
                node.func,
                ast.Name
            ):

                raise Exception(
                    "only simple calls supported"
                )


            name = node.func.id



            if name == "program_id":

                return self.emit(
                    "program_id",
                    "i32"
                )



            if name == "arange":

                return self.emit(
                    "lane_id",
                    "i32"
                )



            if name == "load":

                args = [
                    self.expr(x)
                    for x in node.args
                ]

                return self.emit(
                    "load",
                    "f32",
                    *args
                )



            if name == "store":

                args = [
                    self.expr(x)
                    for x in node.args
                ]

                self.emit(
                    "store",
                    None,
                    *args
                )

                return None



        raise Exception(
            ast.dump(node)
        )