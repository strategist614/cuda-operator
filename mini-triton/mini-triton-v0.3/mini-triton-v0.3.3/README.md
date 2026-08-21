# Mini Triton v0.3.3

Mini Triton v0.3.3 用一个独立原型探索 tensor scalarization 和虚拟寄存器。Frontend 先构造固定 256 元素的 Tensor IR，RegisterLoweringPass 再追加 scalarize/allocate 标记，backend 最后输出两次 load、一次 FP32 add 和一次 store 的 PTX 风格文本。

当前实现的重点是 Register IR 的概念，不是完整 GPU codegen。输出缺少 PTX module、kernel ABI、寄存器声明、线程/地址映射和输入指针，不能由 CUDA Driver API 执行。

## 本版本新增内容

- `Register`：表示带 dtype 的虚拟寄存器。
- `RegisterMapping`：计算某线程负责的逻辑元素编号。
- `RegisterLoweringPass`：追加 tensor scalarization 与 register allocation operation。
- Backend 输出 `%f0`、`%f1`、`%f2` 形式的 PTX 风格浮点寄存器操作。

## 与 v0.3.2 的关系

| 项目 | v0.3.2 | v0.3.3 |
| --- | --- | --- |
| 主要关注点 | Thread→address lowering | Tensor→register lowering |
| ThreadMapping/LaunchConfig | 有 | RegisterMapping 仅定义，未接入 |
| AddressLoweringPass | 有 | 无 |
| RegisterLoweringPass | 无 | 有 |
| Backend 草图 | thread id、地址乘法、一次 load | 两次 load、一次 add、一次 store |
| 完整 pipeline | 不完整 | 同样不完整 |

旧 README 把 pipeline 写成 Tensor→Layout→Thread→Address→Register→PTX，但当前 `compiler.py` 实际没有导入或执行 layout、thread、address passes。v0.3.3 是单独验证 register lowering 结构的原型，并不是把 v0.3.2 全部阶段真正串联后的版本。

## 实际编译流程

```text
Python DSL
    ↓ Frontend
Tensor IR（固定 tensor<256xf32>）
    ↓ RegisterLoweringPass
scalarize_tensor + allocate_register markers
    ↓ PTXBackend
IR 注释 + virtual-register PTX 草图
```

实际入口：

```python
def compile_kernel(src):
    ir = Frontend().compile(src)
    RegisterLoweringPass().run(ir)
    return ir, PTXBackend().emit(ir)
```

## 文件结构

| 文件 | 作用 |
| --- | --- |
| `register_ir.py` | 定义 Register 与 RegisterMapping |
| `ir.py` | 定义 `Value`、`Op`、`FunctionIR` |
| `frontend.py` | Python AST → 固定 shape 的 Tensor IR |
| `passes.py` | 实现 RegisterLoweringPass |
| `backend.py` | 输出 IR 注释和虚拟寄存器 PTX 草图 |
| `compiler.py` | 运行 frontend、register lowering 与 backend |
| `demo.py` | 编译并打印一个 tensor add 示例 |

`__pycache__/` 是 Python 自动生成的字节码缓存，不属于编译器源码。

## Demo DSL

```python
def add_kernel(X, Y, Z):
    offsets = arange(256)
    x = load(X, offsets)
    y = load(Y, offsets)
    z = x + y
    store(Z, offsets, z)
```

目标语义是：

```text
Z[0:256] = X[0:256] + Y[0:256]
```

当前 DSL 没有 program id、N 或 mask，只描述一个固定 tensor block。后续 backend 草图也没有保留 X/Y/Z 的区别，因此还不能实现这个目标语义。

## IR 结构

### Value

```python
@dataclass
class Value:
    name: str
    ty: object
```

Frontend 用 operation 列表当前长度生成 `%0`、`%1` 等名称。

### Op

```python
@dataclass
class Op:
    opcode: str
    result: object
    args: tuple
```

`Op.__str__()` 会打印可选结果、opcode 和 operands。

### FunctionIR

FunctionIR 保存函数名、参数和 operation 列表。`dump()` 是项目自定义文本格式，不是 MLIR、LLVM IR 或官方 Triton IR。

## Frontend

Frontend 选择源码中的第一个 Python 函数，并把参数 X、Y、Z 注册为 `ptr`。

当前 lowering：

| Python/DSL | IR |
| --- | --- |
| 函数参数 | `Value(name, "ptr")` |
| `arange(...)` | `tensor_value : tensor<256xf32>` |
| `load(ptr, offsets)` | `tensor_load : tensor<256xf32>` |
| 任意二元表达式 | `add : tensor` |
| `store(ptr, offsets, value)` | `store`，但仍产生一个 `ty=None` 的 Value |

存在几项重要的简化：

- `arange()` 实参被忽略，shape 固定为 256。
- offsets 被错误标成 `tensor<256xf32>`，概念上索引更适合使用整数类型。
- 所有 `ast.BinOp` 都被当成 add，没有检查实际运算符。
- load 的结果类型固定为 `tensor<256xf32>`。
- `emit()` 无论 ty 是否为 None 都会创建 Value，所以 store 会显示为 `%4:None = store ...`，而不是无结果 operation。
- Frontend 没有 shape/dtype/operand validation。

## Register IR

### Register

```python
@dataclass
class Register:
    name: str
    dtype: str
```

例如：

```text
%f0:f32
%f1:f32
```

这些是虚拟寄存器描述，不是 PTX assembler 中已经声明的真实寄存器。

### RegisterMapping

```python
@dataclass
class RegisterMapping:
    elements_per_thread: int
```

`map(thread_id)` 计算该线程负责的连续元素编号：

```python
base = thread_id * elements_per_thread
return [base + i for i in range(elements_per_thread)]
```

若 `elements_per_thread=2`：

```text
thread 0 → [0, 1]
thread 1 → [2, 3]
...
```

但是当前 compiler 和 pass 没有实例化 RegisterMapping，它只是一个尚未接入 pipeline 的辅助类，也没有线程范围或 element 越界检查。

## RegisterLoweringPass

该 pass 在原 Tensor IR 的 store 之后追加两条 operation：

```text
scalarize_tensor tensor<256xf32> per-thread values
allocate_register %f0:f32 %f1:f32
```

`scalarize_tensor` 只描述把 tensor 拆成每线程值的意图；它没有遍历或替换原来的 tensor_value/tensor_load/add/store。

`allocate_register` 只声明 `%f0`、`%f1` 两个虚拟输入寄存器。Backend 后面还使用 `%f2` 保存 add 结果，但 RegisterLoweringPass 没有分配 `%f2`。

由于 pass 只在函数尾部追加标记，lowering 后的 IR 仍混合了高层 tensor operations 和 register-level 描述。

## Register IR 输出

demo 的 `ir.dump()` 类似：

```text
func @add_kernel
  arg X:ptr
  arg Y:ptr
  arg Z:ptr
  %0:tensor<256xf32> = tensor_value 256
  %1:tensor<256xf32> = tensor_load X:ptr %0:tensor<256xf32>
  %2:tensor<256xf32> = tensor_load Y:ptr %0:tensor<256xf32>
  %3:tensor = add %1:tensor<256xf32> %2:tensor<256xf32>
  %4:None = store Z:ptr %0:tensor<256xf32> %3:tensor
  scalarize_tensor tensor<256xf32> per-thread values
  allocate_register %f0:f32 %f1:f32
```

这证明 register pass 已被调用，但并不表示 tensor operations 已真正 scalarize。

## Backend PTX 草图

backend 在 IR 注释后追加：

```text
// virtual PTX registers
ld.global.f32 %f0,[addr]
ld.global.f32 %f1,[addr+4]
add.f32 %f2,%f0,%f1
st.global.f32 [addr],%f2
```

它表达“加载两个相邻 FP32、相加、写回第一个地址”的局部寄存器数据流。

但这不等价于 demo 的 `X[offset] + Y[offset]`：

- 两次 load 使用同一个抽象 base `addr`，只是相差 4 bytes。
- 没有分别使用 X 和 Y 的 base pointer。
- Store 也没有使用 Z pointer。
- 没有 thread id、program id 或每线程地址计算。
- 只计算一个输出，不覆盖 256 个元素。

## 为什么当前输出不是可执行 PTX

- 缺少 `.version`、`.target`、`.address_size`。
- 缺少 `.entry` kernel 和 X/Y/Z 参数 ABI。
- `%f0`、`%f1`、`%f2` 没有 `.reg .f32` 声明。
- `addr` 未定义。
- 指令没有分号。
- 没有线程索引、地址 lowering、N 或 mask。
- 没有把 Register IR 与原 Tensor IR operands 关联起来。

因此输出只是 register codegen 草图，不能交给 CUDA Driver API JIT。

## Compiler API

```python
from compiler import compile_kernel

ir, ptx = compile_kernel(src)
print(ir.dump())
print(ptx)
```

`compile_kernel()` 只返回内存中的 IR 与字符串，不创建 output 目录，也不写文件。

## 运行

本版本只依赖 Python 标准库，不需要 GPU：

```bash
cd mini-triton/mini-triton-v0.3/mini-triton-v0.3.3
python demo.py
```

或者：

```bash
conda run -n main python demo.py
```

demo 会打印 Register IR 和 PTX 风格草图，不会初始化 CUDA 或验证数值结果。

## 当前限制

- Tensor shape 和 dtype 全部硬编码，offsets 也被标成 f32。
- 任意 binary operation 都被当成 add。
- store 错误地产生 `ty=None` 的结果 Value。
- RegisterMapping 已定义但未接入 pipeline。
- RegisterLoweringPass 只追加标记，没有重写 tensor operations。
- `%f2` 被 backend 使用但未在 register pass 中分配。
- 没有 layout、thread、address lowering 或 pass 组合。
- Backend 的 addr 和 X/Y/Z pointer 没有关联。
- 输出不是合法 PTX，没有 runtime、正确性测试或 benchmark。

## 下一步建议

1. 修正 Frontend 的 offset dtype 和无结果 store operation。
2. 把 v0.3.2 的 thread/address lowering 与 RegisterLoweringPass 串联。
3. 使用 RegisterMapping 为每线程每个元素建立 SSA/Register 映射。
4. 将 tensor_load/add/store 真正重写为标量 register operations。
5. 为 add 结果分配 `%f2`，并加入类型化虚拟寄存器表。
6. 从 X/Y/Z 参数和线程索引生成正确的 byte address。
7. 生成完整 PTX module、寄存器声明、predicate 和尾部处理。
8. 接入 CUDA Driver runtime，并用 CPU reference 验证向量加。
