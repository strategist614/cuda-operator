# Mini Triton Final（v0.3.8）

`mini-triton-final` 是 v0.3 系列的集成原型，代码内部版本标记为 v0.3.8。它在一个 compiler pipeline 中串联 Tensor IR、BlockedLayout、thread/lane IR、地址 IR 和简单寄存器分配，最后生成 PTX 风格的 codegen 草图。

“Final”表示这一轮教学结构的集成版本，不表示生产可用或已经完成 GPU codegen。当前 backend 仍是模拟实现：没有合法 PTX module、kernel ABI、类型化寄存器、真实 X/Y/Z 地址或 runtime，不能执行 demo 中的向量加法。

## 已集成的阶段

- Python AST → Tensor IR frontend。
- `BlockedLayout(4 warps, 32 threads/warp, 2 elements/thread)`。
- thread id 与 lane id operation。
- 抽象 address operation。
- 简单虚拟寄存器编号。
- PTX 风格 load/add/store 文本。

## 实际编译流程

```text
Python DSL
    ↓ Frontend
Tensor IR（固定 256 元素）
    ↓ LayoutPass
BlockedLayout + layout op
    ↓ ThreadPass
thread_id + lane_id
    ↓ AddressPass
abstract address op
    ↓ RegisterAllocator
SSA-name → virtual-register table
    ↓ PTXBackend
hard-coded PTX simulation
```

对应源码：

```python
ir = Frontend().compile(src)
LayoutPass().run(ir)
ThreadPass().run(ir)
AddressPass().run(ir)
RegisterAllocator().run(ir)
ptx = PTXBackend().emit(ir)
```

与 v0.3.3 不同，本版本确实把 layout、thread、address、register 四个阶段放进同一入口；但这些 pass 主要向 IR 末尾追加 metadata/operation，没有逐步重写并消除上一层 IR。

## 文件结构

| 文件 | 作用 |
| --- | --- |
| `tensor_types.py` | 定义 TensorType 和 RegisterType；当前 pipeline 未使用 |
| `layout.py` | 定义 BlockedLayout 与总线程数计算 |
| `ir.py` | 定义 `Value`、`Op`、`FunctionIR` |
| `frontend.py` | 把受限 Python AST lowering 为固定 Tensor IR |
| `passes.py` | 实现 layout、thread、address 和 register allocation |
| `backend.py` | 生成硬编码的 PTX 风格计算草图 |
| `compiler.py` | 串联全部编译阶段 |
| `demo.py` | 编译并打印 tensor add 的 IR、寄存器表和 PTX 草图 |

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

目标语义为：

```text
Z[0:256] = X[0:256] + Y[0:256]
```

当前没有 program id、N 或 mask，只描述一个固定 tensor block。Backend 也没有保留 X/Y/Z 的独立 base pointer，所以生成文本尚未实现该目标语义。

## IR 数据结构

### Value

```python
@dataclass
class Value:
    name: str
    ty: object
```

Value 保存 SSA-like 名称和类型。

### Op

```python
@dataclass
class Op:
    opcode: str
    result: object
    args: tuple
```

Operation 可包含一个结果和若干 operands。

### FunctionIR

FunctionIR 保存函数名、参数、operation 列表。各 pass 还会动态增加：

- `ir.layout`：BlockedLayout。
- `ir.registers`：SSA 名称到虚拟寄存器名称的映射。

这些字段没有在 FunctionIR dataclass 中正式声明，`dump()` 只通过额外追加的 layout/register_allocate operation 间接显示它们。

## Frontend

Frontend 解析第一个 Python 函数，将所有参数注册为类型字符串 `ptr`，然后处理简单赋值和表达式语句。

| Python/DSL | IR |
| --- | --- |
| 函数参数 | `Value(name, "ptr")` |
| `arange(...)` | `make_tensor : tensor<256xf32>` |
| `load(ptr, offsets)` | `load : tensor` |
| 任意二元表达式 | `add : tensor` |
| `store(ptr, offsets, value)` | 无结果的 `store` |

重要简化：

- `arange()` 实参被忽略，shape 固定为 256。
- offsets 被标为 FP32 tensor，概念上索引应使用整数 dtype。
- load/add 只使用通用 `tensor` 字符串类型。
- 所有 `ast.BinOp` 都会被当成 add。
- 没有 shape、dtype、pointer 或 operand validation。

## TensorType 与 memory space

`tensor_types.py` 定义：

```python
@dataclass
class TensorType:
    shape: tuple
    dtype: str
    memory: str = "global"
```

它可以表达：

```text
tensor<(256,)xf32, global>
```

`RegisterType(dtype)` 也可表示寄存器标量类型。

但 Frontend 仍使用字符串类型，所有 pass/backend 也没有导入 TensorType 或 RegisterType。因此 README 旧版所称的“Memory space model”目前只是数据结构定义，尚未接入实际 IR、验证或 codegen。

## BlockedLayout

LayoutPass 创建：

```python
BlockedLayout(
    warps=4,
    threads_per_warp=32,
    elements_per_thread=2,
)
```

计算结果：

```text
threads        = 4 × 32 = 128
total elements = 128 × 2 = 256
```

并向 IR 追加：

```text
layout blocked(warps=4,threads=128,elements/thread=2)
```

当前没有验证 tensor shape 是否等于 `threads × elements_per_thread`，也没有把 layout 分配传播到每个 tensor Value。

## ThreadPass

ThreadPass 追加两个正规 Value：

```text
%tid:i32 = thread_id
%lane:i32 = lane_id
```

相比早期版本使用普通字符串结果，这里 `%tid` 和 `%lane` 已由 Value 表示。但它们仍被追加在原 tensor store 之后，也没有参与 offsets 或 load/store 地址计算。

概念上：

```text
thread_id ≈ threadIdx.x
lane_id   ≈ threadIdx.x % 32
```

Backend 没有为 lane id 生成指令。

## AddressPass

AddressPass 追加：

```text
%addr:ptr = address tid * elements_per_thread
```

operand `tid * elements_per_thread` 是一个字符串，而不是对 `%tid` 和常量 2 的 SSA 运算。因此 address op 只表达意图，没有数据依赖、base pointer、元素大小或 byte offset。

对 FP32 和每线程 2 个元素，线程首元素的正确字节偏移应为：

```text
first_element = thread_id * 2
byte_offset   = first_element * sizeof(float)
              = thread_id * 8
```

如果加入多个 program，还需要叠加 program/block 的起始 offset。

## RegisterAllocator

RegisterAllocator 遍历所有有结果的 operation，为它们顺序分配通用名称：

```text
%0    → %r0
%1    → %r1
%2    → %r2
%3    → %r3
%tid  → %r4
%lane → %r5
%addr → %r6
```

然后保存到 `ir.registers` 并追加 `register_allocate` operation。

当前 allocator 的限制：

- 不区分 i32、f32、pointer 和 tensor 类型，全部使用 `%rN`。
- 没有 liveness analysis 或寄存器复用。
- 没有处理 register class、width、alignment 或 spilling。
- 没有重写 IR operands/result 使用分配后的寄存器。
- Backend 完全没有读取 `ir.registers`。

因此它更像“SSA 名称编号表”，还不是实际 codegen register allocation。

## 集成后的 IR

demo 的 IR 类似：

```text
func @add_kernel
  arg X:ptr
  arg Y:ptr
  arg Z:ptr
  %0:tensor<256xf32> = make_tensor 256
  %1:tensor = load X:ptr %0:tensor<256xf32>
  %2:tensor = load Y:ptr %0:tensor<256xf32>
  %3:tensor = add %1:tensor %2:tensor
  store Z:ptr %0:tensor<256xf32> %3:tensor
  layout blocked(warps=4,threads=128,elements/thread=2)
  %tid:i32 = thread_id
  %lane:i32 = lane_id
  %addr:ptr = address tid * elements_per_thread
  register_allocate ('%0', '%r0') ... ('%addr', '%r6')
```

它同时包含 Tensor IR、layout/thread/address 标记和 register allocation 表，展示了阶段集成；但高层 operation 尚未被逐层替换成最终低层 IR。

## Backend PTX 模拟

Backend 输出：

```text
mov.u32 %r1,%tid.x;
mul.lo.u32 %r2,%r1,2;
ld.global.f32 %f1,[addr+%r2];
ld.global.f32 %f2,[addr+%r2+4];
add.f32 %f3,%f1,%f2;
st.global.f32 [addr],%f3;
```

它模拟每线程读取两个相邻 FP32、相加并写回，但存在关键语义问题：

- `%r2 = thread_id × 2` 是元素索引，却被直接当作字节偏移；FP32 应乘 8 得到线程首元素字节偏移。
- 两次 load 使用同一抽象 `addr`，不是分别从 X 与 Y 加载同一 offset。
- Store 固定写 `[addr]`，没有加 `%r2`，所有线程会写同一地址并产生 data race。
- 没有使用 Z base pointer。
- Backend 的 `%r1/%r2` 与 `ir.registers` 分配表无关。
- Lane id、layout、register allocation 都没有真正影响指令生成。

所以它不等价于 `Z[i] = X[i] + Y[i]`。

## 为什么输出仍不是合法 PTX

- 缺少 `.version`、`.target`、`.address_size`。
- 缺少 `.entry` kernel 与 X/Y/Z 参数 ABI。
- `%r*`、`%f*` 没有寄存器声明。
- `addr` 未定义。
- 没有 program id、N、mask 或 predicate。
- 地址单位和 store 地址错误。
- 没有将 IR 与 register allocation 结果传给 backend。

因此不能把 backend 字符串交给 CUDA Driver API JIT。

## Compiler API

```python
from compiler import compile_kernel

ir, ptx = compile_kernel(src)
print(ir.dump())
print(ir.registers)
print(ptx)
```

`compile_kernel()` 只返回内存中的 IR 和字符串，不创建 output 目录，也不写文件。

## 运行

本版本只依赖 Python 标准库，不要求 GPU：

```bash
cd mini-triton/mini-triton-v0.3/mini-triton-final
python demo.py
```

或者：

```bash
conda run -n main python demo.py
```

demo 会打印集成 IR、寄存器映射和 PTX 模拟，不会初始化 CUDA 或检查数值结果。

## 当前限制

- TensorType、RegisterType 和 memory space model 未接入 pipeline。
- Tensor shape/dtype 和 layout 全部固定。
- 任意 binary operation 都被当成 add。
- Pass 只追加 operation，没有真正逐层 lowering/rewriting。
- Thread/lane/address 出现在原 store 之后，没有参与原数据流。
- Address operand 是字符串，不是 SSA 表达式。
- RegisterAllocator 不区分类型、不重写 IR，backend 也不使用其结果。
- Backend 地址缩放、X/Y/Z base 和 store offset 都不正确。
- 输出不是合法 PTX，没有 runtime、测试或 benchmark。

## 下一步建议

1. 让 Frontend 使用 TensorType/RegisterType，而不是字符串类型。
2. 增加 program id、N、mask 和多 block 语义。
3. 把 layout/thread/address/register 定义为正式的分层 IR。
4. 让每个 pass 重写并消费上一层 operation，而不是只在尾部追加标记。
5. 使用类型化 SSA operation 计算元素索引和字节地址。
6. 实现按类型分类的 register allocation，并重写所有 operands。
7. 让 backend 完全由 lowered IR 驱动，移除硬编码指令。
8. 生成合法 PTX module、kernel ABI、predicate load/store。
9. 接入 CUDA Driver runtime，并加入 CPU reference correctness tests。
