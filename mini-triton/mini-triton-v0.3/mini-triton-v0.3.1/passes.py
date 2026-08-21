from ir import Op
from thread_ir import ThreadMapping,LaunchConfig

class LayoutLoweringPass:
    def run(self,ir):
        ir.mapping=ThreadMapping(128,2)
        ir.launch=LaunchConfig(128,4)
        ir.ops.append(Op("thread_mapping",None,
                         ("128 threads","2 elements/thread")))
        return ir


class ThreadLoweringPass:
    def run(self,ir):
        ir.ops.append(Op("thread_id","%tid",()))
        ir.ops.append(Op("lane_id","%lane",()))
        return ir
