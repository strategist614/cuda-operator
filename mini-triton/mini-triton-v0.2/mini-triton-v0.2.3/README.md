# Mini Triton v0.2.3

Mini Triton v0.2.3 是 v0.2 系列的收尾版本。它保留 tensor-level IR 和 shape/dtype 检查，并把编译流程明确拆成 Frontend、Type/Shape Pass、Simplify Pass 和 Backend 四个阶段。

“Final”表示 v0.2 教学阶段的结构收尾，并不表示已经生成可执行 GPU 程序。当前 PTX backend 仍只输出 opcode 注释，SimplifyPass 也是空实现。

## 本版本整合内容

- Tensor IR：`arange`、`tensor_load`、`tensor_add`、`tensor_store`。
- `TensorType` 与 `PointerType`。
- 从 Python 函数签名提取 pointer 参数。
- 基于 offsets 的 load shape inference。
- `TypeShapePass` 对 tensor add 执行 shape/dtype 检查。
- `SimplifyPass` 优化阶段接口。
- PTX backend 接口。

## 相比 v0.2.2 的变化

| 项目 | v0.2.2 | v0.2.3 |
| --- | --- | --- |
| Tensor shape 容器 | list，例如 `[256]` | tuple，例如 `(256,)` |
| 检查 pass | `TypeCheckPass` | `TypeShapePass` |
| load pointer/offset 检查 | 有 | 移除，只检查 tensor add |
| 优化 pass | 无 | 新增空实现 `SimplifyPass` |
| 编译输出文件 | 自动写 `.tir/.ptx` | 不写文件，只返回 IR/PTX |
| compiler 参数 | `compile_kernel(src, name)` | `compile_kernel(src)` |
| backend 标识 | `v0.2.2` | `v0.2 final` |

因此 v0.2.3 的 pipeline 更完整，但类型检查覆盖范围并不是 v0.2.2 的严格超集。

## 编译流程

```text
Python DSL source
        ↓ Frontend
Tensor IR + inferred types
        ↓ TypeShapePass
Shape/dtype-checked IR
        ↓ SimplifyPass（当前不修改 IR）
Optimized IR
        ↓ PTXBackend
Placeholder PTX text
```

实际入口：

```python
def compile_kernel(src):
    ir = Frontend().compile(src)
    TypeShapePass().run(ir)
    SimplifyPass().run(ir)
    return ir, PTXBackend().emit(ir)
```

## 文件结构

| 文件 | 作用 |
| --- | --- |
| `tensor_types.py` | 定义 tuple shape 的 `TensorType` 与通用 `PointerType` |
| `ir.py` | 定义 `Value`、`Op`、`FunctionIR` |
| `frontend.py` | Python AST → Tensor IR |
| `passes.py` | 定义 `TypeShapePass` 与 `SimplifyPass` |
| `backend.py` | 遍历 IR，当前只输出 opcode 注释 |
| `compiler.py` | 顺序运行 frontend、passes 和 backend |
| `demo.py` | 编译并打印一个 256 元素 tensor add |

`__pycache__/` 是 Python 自动生成的字节码缓存，不属于编译器逻辑。

## Demo DSL

```python
def add_kernel(X, Y, Z):
    offsets = arange(256)
    x = load(X, offsets)
    y = load(Y, offsets)
    z = x + y
    store(Z, offsets, z)
```

逻辑含义为：

```text
offsets = [0, 1, ..., 255]
Z[offsets] = X[offsets] + Y[offsets]
```

当前没有 program id、grid、N 和 mask，只描述一个固定长度为 256 的 tensor block。

## 类型系统

### TensorType

```python
@dataclass
class TensorType:
    dtype: str
    shape: tuple
```

Frontend 为 `arange(256)` 构造：

```python
TensorType("i32", (256,))
```

load 复制 offsets 的 shape，并把 dtype 设为 FP32：

```python
TensorType("f32", off.ty.shape)
```

类型打印结果仍采用：

```text
tensor<(256,)xi32>
tensor<(256,)xf32>
```

### PointerType

函数的 X、Y、Z 参数都被建模为通用 `ptr`。PointerType 没有 dtype 字段，也不能区分地址空间或读写属性。

## IR

`Value` 表示带类型的 SSA-like value，`Op` 保存 opcode、结果和 operands，`FunctionIR` 保存函数参数及 operation 列表。

根据当前打印逻辑，demo IR 类似：

```text
func @add_kernel
  arg X:ptr
  arg Y:ptr
  arg Z:ptr
  %0:tensor<(256,)xi32> = arange 256
  %1:tensor<(256,)xf32> = tensor_load X:ptr %0:tensor<(256,)xi32>
  %2:tensor<(256,)xf32> = tensor_load Y:ptr %0:tensor<(256,)xi32>
  %3:tensor<(256,)xf32> = tensor_add %1:tensor<(256,)xf32> %2:tensor<(256,)xf32>
  tensor_store Z:ptr %0:tensor<(256,)xi32> %3:tensor<(256,)xf32>
```

这是项目自定义的 dump 格式，不是官方 Triton IR 或 MLIR。

## Frontend

Frontend 使用 `ast.parse()` 找到第一个函数，把所有函数参数注册为 PointerType，再顺序编译简单赋值和表达式语句。

| Python/DSL | Tensor IR |
| --- | --- |
| 函数参数 | pointer `Value` |
| `arange(size)` | `tensor<(size,)xi32>` |
| `load(ptr, offsets)` | `tensor<offsets.shape × f32>` |
| 二元表达式 | `tensor_add` |
| `store(ptr, offsets, value)` | `tensor_store` |

当前实现对所有 `ast.BinOp` 都生成 `tensor_add`，没有检查原运算符是否真的是 `+`。`arange` 的 size 必须是整数字面量。

## TypeShapePass

该 pass 只处理 `tensor_add(a, b)`：

- shape 不一致时抛出 `TypeError("shape mismatch")`。
- dtype 不一致时抛出 `TypeError("dtype mismatch")`。

它没有验证 operand 是否为 TensorType，也不再像 v0.2.2 那样检查 `tensor_load` 的 pointer 和 offset 类型；`tensor_store` 同样未被检查。

## SimplifyPass

```python
class SimplifyPass:
    def run(self, ir):
        return ir
```

当前只定义优化阶段边界，不会改变 IR。后续可在这里实现常量折叠、冗余 load 消除、代数化简或死代码消除。

## Backend 当前状态

backend 输出如下形式：

```text
// Mini Triton v0.2 final
// arange
// tensor_load
// tensor_load
// tensor_add
// tensor_store
```

这不是合法的完整 PTX module：它没有版本/架构 header、kernel entry、参数、寄存器或指令，不能由 CUDA Driver API 加载执行。

## Compiler API

```python
from compiler import compile_kernel

ir, ptx = compile_kernel(src)
print(ir.dump())
print(ptx)
```

v0.2.3 只返回内存中的 IR 和 PTX 字符串，不创建 `output/`，也不写 `.tir` 或 `.ptx` 文件。如需保存结果，应由调用方显式写入。

## 运行

本版本只使用 Python 标准库，不需要 GPU：

```bash
cd mini-triton/mini-triton-v0.2/mini-triton-v0.2.3
python demo.py
```

也可以运行：

```bash
conda run -n main python demo.py
```

demo 只打印 IR 和占位 PTX，不会启动 CUDA kernel 或验证数值结果。

## 当前限制

- 所有函数参数都被视为无元素类型的 pointer。
- 只支持简单赋值、表达式语句、`arange`、`load`、`store` 和 tensor add。
- 任意二元运算都会被错误地当成 tensor add。
- `arange` size 仅支持整数字面量。
- load 固定返回 FP32 tensor。
- TypeShapePass 只检查 add 的 shape/dtype，未检查 load/store。
- SimplifyPass 是空实现。
- 没有 program/grid、mask、广播、归约、多维 layout 或控制流。
- Frontend 实例重复编译时不会重置内部状态。
- Backend 只输出注释，不生成可执行 PTX。
- 没有输出文件、runtime、单元测试、正确性检查或 benchmark。

## v0.2 系列之后的建议

1. 为 DSL 建立明确的函数参数和 scalar/pointer/tensor 类型语法。
2. 合并并补全 load、store、binary op 的 type/shape verification。
3. 增加 program id、mask 和 tensor layout，把 block 语义映射到 CUDA threads/warps。
4. 让 SimplifyPass 实现至少一个可测试的真实优化。
5. 为所有 SSA value 分配 PTX 寄存器并实现 kernel ABI。
6. Lower tensor load/add/store 为真实 PTX，并接入 CUDA Driver runtime。
7. 建立 frontend、pass、backend snapshot 和 GPU correctness 测试。
