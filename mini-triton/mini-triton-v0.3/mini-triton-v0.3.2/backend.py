
class PTXBackend:

    def emit(self,ir):

        out=[
            "// Mini Triton v0.3.2",
            "// Thread -> Address lowering",
            "// "+str(ir.launch),
            ""
        ]

        out += [
            "// "+str(op)
            for op in ir.ops
        ]

        out += [
            "",
            "mov.u32 %r1, %tid.x;",
            "mul.lo.u32 %r2, %r1, 2;",
            "ld.global.f32 %f1, [addr+%r2];"
        ]

        return "\n".join(out)
