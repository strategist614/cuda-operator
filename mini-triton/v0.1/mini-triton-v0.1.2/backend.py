class PTXBackend:
    def emit(self, ir):
        out = [
            ".version 7.0",
            ".target sm_70",
            ".address_size 64",
            "",
            f".visible .entry {ir.name}()",
            "{"
        ]

        for op in ir.ops:
            if op.opcode=="program_id":
                out.append("    mov.u32 %r1, %ctaid.x;")

            elif op.opcode=="lane_id":
                out.append("    mov.u32 %r2, %tid.x;")

            elif op.opcode=="add":
                out.append("    add.u32 %r3,%r1,%r2;")

            elif op.opcode=="mul":
                out.append("    mul.lo.u32 %r4,%r1,%r2;")

            elif op.opcode=="load":
                out.append("    // ld.global.f32")

            elif op.opcode=="store":
                out.append("    // st.global.f32")

        out += [
            "    ret;",
            "}"
        ]

        return "\n".join(out)
