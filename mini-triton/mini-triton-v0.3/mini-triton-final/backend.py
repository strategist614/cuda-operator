
class PTXBackend:

    def emit(self,ir):

        out=[
            "// Mini Triton v0.3.8",
            "// PTX simulation",
            f"// layout: {ir.layout}",
            ""
        ]

        out += [
            "mov.u32 %r1,%tid.x;",
            "mul.lo.u32 %r2,%r1,2;",
            "ld.global.f32 %f1,[addr+%r2];",
            "ld.global.f32 %f2,[addr+%r2+4];",
            "add.f32 %f3,%f1,%f2;",
            "st.global.f32 [addr],%f3;"
        ]

        return "\n".join(out)
