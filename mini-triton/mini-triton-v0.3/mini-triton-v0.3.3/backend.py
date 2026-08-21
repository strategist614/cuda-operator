
class PTXBackend:

    def emit(self,ir):

        out=[
            "// Mini Triton v0.3.3",
            "// Register lowering"
        ]

        for op in ir.ops:
            out.append("// "+str(op))

        out += [
            "",
            "// virtual PTX registers",
            "ld.global.f32 %f0,[addr]",
            "ld.global.f32 %f1,[addr+4]",
            "add.f32 %f2,%f0,%f1",
            "st.global.f32 [addr],%f2"
        ]

        return "\n".join(out)
