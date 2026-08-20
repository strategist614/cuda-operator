# Mini Triton v0.2.2

Mini Triton v0.2.2 在 v0.2 Tensor IR 的基础上加入了独立的类型检查 pass。Frontend 先推导 tensor shape 和 dtype，`TypeCheckPass` 再验证 tensor 加法与加载操作，检查通过后才交给 PTX backend。

当前 backend 仍是占位实现，只把 IR opcode 输出为注释，并未生成可执行 PTX。本版本主要用于学习编译器中的类型表示、局部 shape inference、validation pass 和 pipeline 组织。

## 本版本新增内容

- 新增 `ScalarType` 类型定义，为后续标量参数预留接口。
- `arange(size)` 推导出 `tensor<[size]xi32>`。
- `load(ptr, offsets)` 根据 offsets shape 推导 FP32 tensor 类型。
- `tensor_add` 保留输入 tensor 类型。
- `TypeCheckPass` 验证 tensor add 的 shape 和 dtype。
- `TypeCheckPass` 验证 load 的 base 是 pointer、offset 是 tensor。
- `compile_kernel()` 正式在 frontend 与 backend 之间运行类型检查。
- 编译结果写入 `output/<name>.tir` 和 `output/<name>.ptx`。

## 编译流程

```text
Python DSL
    ↓ ast.parse()
Python AST
    ↓ Frontend
Tensor IR + inferred types
    ↓ TypeCheckPass
Validated Tensor IR
    ├── dump → output/<name>.tir
    ↓ PTXBackend
Placeholder PTX → output/<name>.ptx
```

如果类型或 shape 检查失败，编译会在 backend 运行之前抛出 `TypeError`，不会继续生成新的输出文件。

## 文件结构

| 文件或目录 | 作用 |
| --- | --- |
| `tensor_types.py` | 定义 `TensorType`、`PointerType` 和 `ScalarType` |
| `ir.py` | 定义 `Value`、`Op`、`FunctionIR` 及文本 dump |
| `frontend.py` | Python AST → Tensor IR，并执行局部类型推导 |
| `passes.py` | 实现 `TypeCheckPass` |
| `backend.py` | 遍历已检查的 IR；当前只生成 opcode 注释 |
| `compiler.py` | 串联 frontend、type check、dump 和 backend |
| `demo.py` | 编译一个 256 元素的 tensor add 示例 |
| `output/add_kernel.tir` | demo 生成的 Tensor IR 快照 |
| `output/add_kernel.ptx` | demo 生成的占位 PTX 快照 |

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

它描述一个 tensor block 内的向量加法：

```text
offsets = [0, 1, ..., 255]
Z[offsets] = X[offsets] + Y[offsets]
```

当前没有 `program_id`、grid、N 或 mask，所以 DSL 只表示从 0 开始的 256 个元素，不能覆盖任意长度输入或安全处理尾部。

## 类型系统

### TensorType

```python
@dataclass
class TensorType:
    dtype: str
    shape: list
```

`dtype` 当前使用 `i32` 或 `f32` 字符串，shape 使用 Python list。例如：

```text
tensor<[256]xi32>
tensor<[256]xf32>
```

### PointerType

函数签名中的每个参数都会被 Frontend 设为通用 `ptr`。PointerType 尚未携带元素 dtype、地址空间或 const 信息。

### ScalarType

`ScalarType(dtype)` 能表示一个标量类型，但当前 Frontend、IR 和 demo 都没有使用它。函数参数仍全部按 pointer 处理。

## Frontend 与 shape inference

Frontend 选择源码中的第一个函数，依次注册参数并编译函数体。当前 lowering 规则为：

| DSL 表达式 | 结果 |
| --- | --- |
| 函数参数 | `Value(arg_name, PointerType())` |
| `arange(size)` | `TensorType("i32", [size])` |
| `load(ptr, offsets)` | `TensorType("f32", offsets.shape)` |
| `a + b` | `tensor_add`，结果类型直接继承 `a.ty` |
| `store(ptr, offsets, value)` | 无返回值的 `tensor_store` |

这里的 shape inference 是局部规则：load 从 offsets 复制 shape，add 从左操作数复制类型。Frontend 本身不会验证左右操作数是否兼容，兼容性由后续 pass 检查。

## TypeCheckPass

`TypeCheckPass.run()` 顺序遍历所有 operation。

对 `tensor_add(a, b)` 检查：

- `a.ty.shape == b.ty.shape`，否则抛出 `TypeError("tensor shape mismatch")`。
- `a.ty.dtype == b.ty.dtype`，否则抛出 `TypeError("tensor dtype mismatch")`。

对 `tensor_load(ptr, offset)` 检查：

- `ptr.ty` 必须是 `PointerType`。
- `offset.ty` 必须是 `TensorType`。

当前 pass 没有检查 `tensor_store`，也没有先验证 add operand 一定是 TensorType。因此对于结构异常的 IR，可能出现属性访问错误，而不是统一的编译诊断。

## 生成的 Tensor IR

demo 会生成：

```text
func @add_kernel
  arg X:ptr
  arg Y:ptr
  arg Z:ptr
  %0:tensor<[256]xi32> = arange 256
  %1:tensor<[256]xf32> = tensor_load X:ptr %0:tensor<[256]xi32>
  %2:tensor<[256]xf32> = tensor_load Y:ptr %0:tensor<[256]xi32>
  %3:tensor<[256]xf32> = tensor_add %1:tensor<[256]xf32> %2:tensor<[256]xf32>
  tensor_store Z:ptr %0:tensor<[256]xi32> %3:tensor<[256]xf32>
```

这是项目自定义的可读格式，并非 MLIR 或官方 Triton IR。

## Backend 当前状态

`PTXBackend` 只输出：

```text
// Mini Triton v0.2.2
// after type checking
// arange
// tensor_load
// tensor_load
// tensor_add
// tensor_store
```

虽然文件扩展名是 `.ptx`，内容还没有 PTX header、kernel entry、参数 ABI、寄存器、地址计算和可执行指令，不能由 CUDA Driver API 加载。

## Compiler API

```python
from compiler import compile_kernel

ir, ptx = compile_kernel(src, "add_kernel")
```

编译入口依次执行：

1. `Frontend().compile(src)`。
2. `TypeCheckPass().run(ir)`。
3. 创建当前工作目录下的 `output/`。
4. 写入 `output/<name>.tir`。
5. 执行 `PTXBackend().emit(ir)`。
6. 写入 `output/<name>.ptx`。
7. 返回 `(ir, ptx)`。

`name` 只控制输出文件名；IR 函数名取自 DSL 源码。

## 运行

本版本只依赖 Python 标准库，不要求 GPU：

```bash
cd mini-triton/mini-triton-v0.2/mini-triton-v0.2.2
python demo.py
```

也可以使用仓库当前 Conda 环境：

```bash
conda run -n main python demo.py
```

运行后会打印 Tensor IR 和占位 PTX，并更新 `output/add_kernel.tir`、`output/add_kernel.ptx`。

## 当前限制

- 所有 kernel 参数均被视为无元素类型的 pointer。
- ScalarType 已定义但尚未使用。
- 只支持简单赋值、`arange`、`load`、`store` 和 `+`。
- `arange` size 必须是整数字面量。
- load 固定返回 FP32 tensor。
- 没有 program/grid、mask、broadcast、reduce、多维 tensor 或控制流。
- TypeCheckPass 只覆盖 tensor add 和 tensor load，未检查 store。
- Frontend 实例重复编译时不会重置内部 IR 状态。
- Backend 只生成注释，不是可执行 PTX。
- 没有 runtime、负例单元测试、正确性验证或性能测试。

## 下一步建议

1. 为 scalar 与 `ptr<dtype>` 增加真实参数类型语法。
2. 补全 store、arange 和所有 operand 的类型检查。
3. 引入 program id、mask 和任意 N 的执行语义。
4. 设计 tensor block 到 CUDA threads/warps 的 layout。
5. 实现真实 PTX ABI、寄存器分配、地址计算和指令 lowering。
6. 增加类型错误单元测试和 CUDA runtime 正确性测试。
