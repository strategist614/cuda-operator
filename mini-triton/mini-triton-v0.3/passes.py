from ir import Op
from layout import BlockedLayout


class LayoutPass:

    def run(self, ir):

        new_ops = []

        for op in ir.ops:

            new_ops.append(op)

            if op.opcode == "make_tensor":

                old_type = op.result.ty

                old_type.layout = BlockedLayout(
                    threads=128,
                    elements_per_thread=2
                )


        ir.ops = new_ops

        return ir



class ThreadMappingPass:

    def run(self, ir):

        new_ops = []

        for op in ir.ops:

            new_ops.append(op)

            if op.opcode == "make_tensor":

                new_ops.append(
                    Op(
                        "thread_mapping",
                        None,
                        (
                            "threads=128",
                            "elements_per_thread=2"
                        )
                    )
                )


        ir.ops = new_ops

        return ir