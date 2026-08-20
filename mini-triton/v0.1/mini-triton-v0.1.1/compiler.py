from passes import ConstantFoldPass

class Compiler:
    def compile(self, ir):
        return ConstantFoldPass().run(ir)
