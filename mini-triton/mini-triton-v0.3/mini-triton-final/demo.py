
from compiler import compile_kernel


src='''
def add_kernel(X,Y,Z):

    offsets=arange(256)

    x=load(X,offsets)

    y=load(Y,offsets)

    z=x+y

    store(Z,offsets,z)
'''


ir,ptx=compile_kernel(src)

print("===== IR =====")
print(ir.dump())

print("\n===== Registers =====")
print(ir.registers)

print("\n===== PTX =====")
print(ptx)
