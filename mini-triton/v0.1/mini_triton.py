from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


# ============================================================
# 1. Mini Triton IR
# ============================================================

@dataclass(frozen=True)
class Value:
    """
    SSA value.

    比如：
        %0 : i32
        %1 : f32
    """
    name: str
    ty: str


@dataclass(frozen=True)
class Argument:
    """
    Kernel parameter.

    X/Y/Z : ptr<f32>
    N     : i32
    """
    name: str
    ty: str


@dataclass
class Op:
    """
    一条 IR operation。

    例如：

        %3 = add %1, %2

    opcode = "add"
    result = %3
    args   = (%1, %2)
    """
    opcode: str
    result: Value | None
    args: tuple[Any, ...]


@dataclass
class FunctionIR:
    name: str
    args: list[Argument]
    ops: list[Op]

    def __str__(self) -> str:
        lines = []

        arg_text = ", ".join(
            f"{arg.name}: {arg.ty}"
            for arg in self.args
        )

        lines.append(f"func @{self.name}({arg_text}) {{")

        for op in self.ops:
            if op.result is not None:
                lhs = f"{op.result.name}:{op.result.ty} = "
            else:
                lhs = ""

            def show(x):
                if isinstance(x, (Value, Argument)):
                    return x.name
                return str(x)

            args = ", ".join(show(x) for x in op.args)

            lines.append(
                f"  {lhs}{op.opcode} {args}".rstrip()
            )

        lines.append("}")

        return "\n".join(lines)


# ============================================================
# 2. IR Builder
# ============================================================

class IRBuilder:
    """
    自动产生 SSA value：

        %0
        %1
        %2
        ...
    """

    def __init__(self):
        self.ops: list[Op] = []
        self.next_id = 0

    def emit(
        self,
        opcode: str,
        ty: str | None = None,
        *args,
    ) -> Value | None:

        result = None

        if ty is not None:
            result = Value(
                name=f"%{self.next_id}",
                ty=ty,
            )
            self.next_id += 1

        self.ops.append(
            Op(
                opcode=opcode,
                result=result,
                args=args,
            )
        )

        return result


# ============================================================
# 3. Python AST -> Mini Triton IR
# ============================================================

class Frontend:
    """
    我们自己的 Mini Triton frontend。

    目前只支持：

        program_id(0)
        arange(BLOCK)

        load(...)
        store(...)

        +
        *

        <
    """

    def __init__(self, block: int):
        self.block = block
        self.builder = IRBuilder()

        # symbol table
        self.env: dict[str, Any] = {}

    def compile(self, source: str) -> FunctionIR:
        tree = ast.parse(source)

        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        ]

        if len(functions) != 1:
            raise SyntaxError(
                "v0.1 requires exactly one kernel function"
            )

        fn = functions[0]

        # 第一版 ABI 暂时固定死
        arguments = [
            Argument("X", "ptr<f32>"),
            Argument("Y", "ptr<f32>"),
            Argument("Z", "ptr<f32>"),
            Argument("N", "i32"),
        ]

        self.env = {
            arg.name: arg
            for arg in arguments
        }

        # BLOCK 是 compile-time constant
        self.env["BLOCK"] = self.block

        for stmt in fn.body:
            self.compile_stmt(stmt)

        return FunctionIR(
            name=fn.name,
            args=arguments,
            ops=self.builder.ops,
        )

    # --------------------------------------------------------

    def compile_stmt(self, node):
        if isinstance(node, ast.Assign):

            if (
                len(node.targets) != 1
                or not isinstance(node.targets[0], ast.Name)
            ):
                raise SyntaxError(
                    "only simple assignments are supported"
                )

            name = node.targets[0].id

            value = self.compile_expr(node.value)

            self.env[name] = value
            return

        if isinstance(node, ast.Expr):
            self.compile_expr(node.value)
            return

        raise SyntaxError(
            f"unsupported statement: {ast.dump(node)}"
        )

    # --------------------------------------------------------

    def compile_expr(self, node):

        # -----------------------
        # variable
        # -----------------------

        if isinstance(node, ast.Name):

            if node.id not in self.env:
                raise NameError(node.id)

            return self.env[node.id]

        # -----------------------
        # literal
        # -----------------------

        if isinstance(node, ast.Constant):
            return node.value

        # -----------------------
        # a + b
        # a * b
        # -----------------------

        if isinstance(node, ast.BinOp):

            lhs = self.compile_expr(node.left)
            rhs = self.compile_expr(node.right)

            # Python integer ->
            # explicit IR constant
            if isinstance(lhs, int):
                lhs = self.builder.emit(
                    "const",
                    "i32",
                    lhs,
                )

            if isinstance(rhs, int):
                rhs = self.builder.emit(
                    "const",
                    "i32",
                    rhs,
                )

            if isinstance(node.op, ast.Add):
                opcode = "add"

            elif isinstance(node.op, ast.Mult):
                opcode = "mul"

            else:
                raise SyntaxError(
                    "v0.1 only supports + and *"
                )

            # 极简 type inference
            if (
                getattr(lhs, "ty", None) == "f32"
                or getattr(rhs, "ty", None) == "f32"
            ):
                ty = "f32"
            else:
                ty = "i32"

            return self.builder.emit(
                opcode,
                ty,
                lhs,
                rhs,
            )

        # -----------------------
        # a < b
        # -----------------------

        if isinstance(node, ast.Compare):

            if (
                len(node.ops) != 1
                or not isinstance(node.ops[0], ast.Lt)
            ):
                raise SyntaxError(
                    "v0.1 only supports <"
                )

            lhs = self.compile_expr(node.left)
            rhs = self.compile_expr(
                node.comparators[0]
            )

            return self.builder.emit(
                "cmp_lt",
                "pred",
                lhs,
                rhs,
            )

        # -----------------------
        # function calls
        # -----------------------

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        ):

            name = node.func.id

            args = [
                self.compile_expr(arg)
                for arg in node.args
            ]

            # program_id(0)
            if name == "program_id":

                if args != [0]:
                    raise SyntaxError(
                        "v0.1 only supports program_id(0)"
                    )

                return self.builder.emit(
                    "program_id",
                    "i32",
                    0,
                )

            # arange(BLOCK)
            if name == "arange":

                if args != [self.block]:
                    raise SyntaxError(
                        "arange() must use BLOCK"
                    )

                # 关键 lowering：
                #
                # arange(BLOCK)
                #       ↓
                # threadIdx.x

                return self.builder.emit(
                    "lane_id",
                    "i32",
                )

            # load(X, offset, mask)
            if name == "load":

                base, offset, mask = args

                return self.builder.emit(
                    "load",
                    "f32",
                    base,
                    offset,
                    mask,
                )

            # store(Z, offset, value, mask)
            if name == "store":

                base, offset, value, mask = args

                self.builder.emit(
                    "store",
                    None,
                    base,
                    offset,
                    value,
                    mask,
                )

                return None

        raise SyntaxError(
            f"unsupported expression: {ast.dump(node)}"
        )


# ============================================================
# 4. Mini Triton IR -> PTX
# ============================================================

class PTXEmitter:

    def __init__(
        self,
        target: str = "sm_70",
        ptx_version: str = "7.0",
    ):
        self.target = target
        self.ptx_version = ptx_version

        # kernel parameter registers
        #
        # X -> rd1
        # Y -> rd2
        # Z -> rd3
        # N -> r1

        self.arg_regs = {
            "X": "%rd1",
            "Y": "%rd2",
            "Z": "%rd3",
            "N": "%r1",
        }

        self.value_regs = {}

        self.next_r = 2
        self.next_rd = 4
        self.next_f = 1
        self.next_p = 1

    # --------------------------------------------------------

    def alloc_reg(self, ty: str) -> str:

        if ty == "i32":
            reg = f"%r{self.next_r}"
            self.next_r += 1
            return reg

        if ty == "f32":
            reg = f"%f{self.next_f}"
            self.next_f += 1
            return reg

        if ty == "pred":
            reg = f"%p{self.next_p}"
            self.next_p += 1
            return reg

        if ty == "i64":
            reg = f"%rd{self.next_rd}"
            self.next_rd += 1
            return reg

        raise ValueError(
            f"unknown type: {ty}"
        )

    # --------------------------------------------------------

    def operand(self, value) -> str:

        if isinstance(value, Argument):
            return self.arg_regs[value.name]

        if isinstance(value, Value):
            return self.value_regs[value.name]

        if isinstance(value, int):
            return str(value)

        raise TypeError(value)

    # --------------------------------------------------------

    def emit(self, fn: FunctionIR) -> str:

        out = [
            f".version {self.ptx_version}",
            f".target {self.target}",
            ".address_size 64",
            "",
            f".visible .entry {fn.name}(",
            "    .param .u64 X,",
            "    .param .u64 Y,",
            "    .param .u64 Z,",
            "    .param .u32 N",
            ")",
            "{",
            "    .reg .pred %p<16>;",
            "    .reg .b32  %r<64>;",
            "    .reg .b64  %rd<64>;",
            "    .reg .f32  %f<64>;",
            "",
            "    ld.param.u64 %rd1, [X];",
            "    ld.param.u64 %rd2, [Y];",
            "    ld.param.u64 %rd3, [Z];",
            "    ld.param.u32 %r1, [N];",
            "",
        ]

        for op in fn.ops:

            if op.result is not None:

                dst = self.alloc_reg(
                    op.result.ty
                )

                self.value_regs[
                    op.result.name
                ] = dst

            else:
                dst = None

            # ----------------------------------------
            # const
            # ----------------------------------------

            if op.opcode == "const":

                out.append(
                    f"    mov.u32 {dst}, {op.args[0]};"
                )

            # ----------------------------------------
            # program_id
            #
            # Triton:
            #
            #     program_id(0)
            #
            # CUDA:
            #
            #     blockIdx.x
            # ----------------------------------------

            elif op.opcode == "program_id":

                out.append(
                    f"    mov.u32 {dst}, %ctaid.x;"
                )

            # ----------------------------------------
            # arange -> threadIdx.x
            # ----------------------------------------

            elif op.opcode == "lane_id":

                out.append(
                    f"    mov.u32 {dst}, %tid.x;"
                )

            # ----------------------------------------
            # integer multiplication
            # ----------------------------------------

            elif op.opcode == "mul":

                lhs, rhs = op.args

                out.append(
                    "    mul.lo.u32 "
                    f"{dst}, "
                    f"{self.operand(lhs)}, "
                    f"{self.operand(rhs)};"
                )

            # ----------------------------------------
            # add
            # ----------------------------------------

            elif op.opcode == "add":

                lhs, rhs = op.args

                if op.result.ty == "f32":
                    suffix = "f32"
                else:
                    suffix = "u32"

                out.append(
                    f"    add.{suffix} "
                    f"{dst}, "
                    f"{self.operand(lhs)}, "
                    f"{self.operand(rhs)};"
                )

            # ----------------------------------------
            # offs < N
            # ----------------------------------------

            elif op.opcode == "cmp_lt":

                lhs, rhs = op.args

                out.append(
                    "    setp.lt.u32 "
                    f"{dst}, "
                    f"{self.operand(lhs)}, "
                    f"{self.operand(rhs)};"
                )

            # ----------------------------------------
            # load
            # ----------------------------------------

            elif op.opcode == "load":

                base, offset, mask = op.args

                byte_offset = self.alloc_reg(
                    "i64"
                )

                address = self.alloc_reg(
                    "i64"
                )

                # float = 4 bytes
                out.append(
                    "    mul.wide.u32 "
                    f"{byte_offset}, "
                    f"{self.operand(offset)}, "
                    "4;"
                )

                out.append(
                    "    add.s64 "
                    f"{address}, "
                    f"{self.operand(base)}, "
                    f"{byte_offset};"
                )

                # masked load false 时先设成 0
                out.append(
                    f"    mov.f32 {dst}, "
                    "0f00000000;"
                )

                out.append(
                    f"    @{self.operand(mask)} "
                    f"ld.global.f32 "
                    f"{dst}, [{address}];"
                )

            # ----------------------------------------
            # store
            # ----------------------------------------

            elif op.opcode == "store":

                base, offset, value, mask = op.args

                byte_offset = self.alloc_reg(
                    "i64"
                )

                address = self.alloc_reg(
                    "i64"
                )

                out.append(
                    "    mul.wide.u32 "
                    f"{byte_offset}, "
                    f"{self.operand(offset)}, "
                    "4;"
                )

                out.append(
                    "    add.s64 "
                    f"{address}, "
                    f"{self.operand(base)}, "
                    f"{byte_offset};"
                )

                out.append(
                    f"    @{self.operand(mask)} "
                    f"st.global.f32 "
                    f"[{address}], "
                    f"{self.operand(value)};"
                )

            else:
                raise ValueError(
                    f"unknown opcode: {op.opcode}"
                )

        out += [
            "",
            "    ret;",
            "}",
        ]

        return "\n".join(out)


# ============================================================
# Public compile API
# ============================================================

def compile_kernel(
    source: str,
    block: int = 256,
):
    frontend = Frontend(
        block=block
    )

    ir = frontend.compile(source)

    ptx = PTXEmitter().emit(ir)

    return ir, ptx