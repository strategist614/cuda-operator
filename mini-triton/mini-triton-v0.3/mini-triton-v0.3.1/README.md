# Mini Triton v0.3.1

Mini Triton v0.3.1 在 v0.3 的 layout 实验基础上，加入显式的线程映射和 launch 配置。编译 pipeline 会为 256 元素 tensor 指定 128 个线程、每线程 2 个元素，并在 IR 中追加 `thread_mapping`、`thread_id` 和 `lane_id` 标记。

本版本仍是编译器结构原型：backend 只把 lowering 后的 IR 输出为注释，不会生成可执行 PTX，也不会启动 GPU。

## 本版本新增内容

- `ThreadMapping`：描述总线程数与每线程处理的元素数。
- `LaunchConfig`：描述每个 block 的线程数与 warp 数。
- `LayoutLoweringPass`：为 FunctionIR 附加 mapping/launch 元数据。
- `ThreadLoweringPass`：追加 `thread_id` 和 `lane_id` operation。
- backend 输出 launch config 与完整 operation 列表。

## 相比 v0.3 的变化

| 项目 | v0.3 | v0.3.1 |
| --- | --- | --- |
| Tensor 类型 | `LayoutTensorType`，类型中持有 BlockedLayout | 使用硬编码字符串，如 `tensor<256xf32>` |
| Layout 表示 | `BlockedLayout(128, 2)` 写入 tensor type | `ThreadMapping(128, 2)` 附加到 FunctionIR |
| Launch 配置 | 无 | `LaunchConfig(128 threads, 4 warps)` |
| Thread IR | 插入一个 `thread_mapping` 描述 | 追加 mapping、thread_id 和 lane_id |
| Backend | dump layout IR 注释 | dump launch config 与 thread IR 注释 |
| GPU codegen | 不支持 | 不支持 |

v0.3.1 增强了线程层的概念，但没有保留 v0.3 的结构化 LayoutTensorType；`layout.py` 中的 BlockedLayout 在当前 pipeline 中也没有被使用。

## 编译流程

```text
Python DSL
    ↓ Frontend
Tensor IR（固定 256 元素）
    ↓ LayoutLoweringPass
ThreadMapping + LaunchConfig + thread_mapping op
    ↓ ThreadLoweringPass
thread_id + lane_id ops
    ↓ PTXBackend
注释形式的 Thread IR dump
```

实际入口：

```python
def compile_kernel(src):
    ir = Frontend().compile(src)
    LayoutLoweringPass().run(ir)
    ThreadLoweringPass().run(ir)
    return ir, PTXBackend().emit(ir)
```

## 文件结构

| 文件 | 作用 |
| --- | --- |
| `ir.py` | 定义 `Value`、`Op`、`FunctionIR` |
| `layout.py` | 定义 BlockedLayout；当前 pipeline 未使用 |
| `thread_ir.py` | 定义 LaunchConfig 与 ThreadMapping |
| `frontend.py` | 把 Python AST lowering 为固定 shape 的 Tensor IR |
| `passes.py` | 实现 layout lowering 与 thread lowering |
| `backend.py` | 输出 launch 和 operation 注释 |
| `compiler.py` | 依次运行 frontend、两个 pass 和 backend |
| `demo.py` | 编译并打印 256 元素 tensor add 示例 |

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

逻辑目标为：

```text
Z[0:256] = X[0:256] + Y[0:256]
```

Frontend 当前没有 program id、N 和 mask，所以只描述一个固定大小的 tensor block，不能覆盖任意长度输入或处理尾部。

## Frontend

Frontend 使用 `ast.parse()` 查找第一个函数，并把每个函数参数注册为类型字符串 `ptr`。

当前 lowering：

| Python/DSL | IR |
| --- | --- |
| 函数参数 | `Value(name, "ptr")` |
| `arange(...)` | `make_tensor : tensor<256xi32>` |
| `load(ptr, offsets)` | `load : tensor<256xf32>` |
| 二元表达式 | `tensor_add : tensor<256xf32>` |
| `store(ptr, offsets, value)` | 无返回值的 `store` |

shape 和 dtype 都是硬编码的：无论 `arange()` 实参是什么，Frontend 都产生 256 个 i32；load 和 binary operation 固定产生 256 个 f32。所有 `ast.BinOp` 都会被当成 tensor add，没有检查实际运算符。

## Thread IR

### LaunchConfig

```python
@dataclass
class LaunchConfig:
    threads_per_block: int
    warps: int
```

本版本固定配置为：

```text
threads_per_block = 128
warps             = 4
```

按每个 warp 32 个线程计算，128 个线程正好是 4 个 warp。当前代码没有验证这两个字段是否一致。

### ThreadMapping

```python
@dataclass
class ThreadMapping:
    threads: int
    elements_per_thread: int
```

固定映射为：

```text
threads             = 128
elements_per_thread = 2
total elements      = 128 × 2 = 256
```

`element_to_thread(e)` 使用整数除法：

```python
thread = e // 2
```

因此逻辑映射为：

```text
element 0, 1     → thread 0
element 2, 3     → thread 1
...
element 254, 255 → thread 127
```

这个类只提供 element→thread 查询，尚未提供“某线程负责哪些 element”的反向接口，也没有检查 e 是否越界。

## LayoutLoweringPass

该 pass 动态给 FunctionIR 增加两个属性：

```python
ir.mapping = ThreadMapping(128, 2)
ir.launch = LaunchConfig(128, 4)
```

随后在 operation 列表末尾追加：

```text
thread_mapping 128 threads 2 elements/thread
```

FunctionIR dataclass 本身没有声明 mapping 和 launch 字段；Python 允许动态添加，但这种信息不会出现在 `ir.dump()` 的函数签名中，只能通过 `ir.mapping`、`ir.launch` 或新增 operation 访问。

## ThreadLoweringPass

该 pass 在 operation 列表末尾追加：

```text
%tid = thread_id
%lane = lane_id
```

这里 `%tid`、`%lane` 是普通字符串，不是由 Frontend SSA allocator 创建的 `Value`。同时 operation 被追加在 tensor_store 之后，没有重写原 tensor load/add/store 去使用 thread id。因此目前它们只是 lowering 阶段标记，还没有真正参与索引或执行。

概念上：

```text
thread_id ≈ CUDA threadIdx.x
lane_id   ≈ threadIdx.x % 32
```

但当前 backend 并没有生成 `%tid.x`、取模或其他 PTX 指令。

## IR 输出

demo 的 `ir.dump()` 会得到类似：

```text
func @add_kernel
  arg X:ptr
  arg Y:ptr
  arg Z:ptr
  %0:tensor<256xi32> = make_tensor 256
  %1:tensor<256xf32> = load X:ptr %0:tensor<256xi32>
  %2:tensor<256xf32> = load Y:ptr %0:tensor<256xi32>
  %3:tensor<256xf32> = tensor_add %1:tensor<256xf32> %2:tensor<256xf32>
  store Z:ptr %0:tensor<256xi32> %3:tensor<256xf32>
  thread_mapping 128 threads 2 elements/thread
  %tid = thread_id
  %lane = lane_id
```

`ir.launch` 另外打印为：

```text
block=128, warps=4
```

## Backend 当前状态

`PTXBackend.emit()` 输出 launch config 和每条 operation 的字符串：

```text
// Mini Triton v0.3.1
// block=128, warps=4
// ... operations ...
```

这不是合法的 PTX module。它没有 `.version`、`.target`、kernel entry、参数 ABI、寄存器、地址计算、global-memory 指令或 launch runtime，不能交给 CUDA Driver API 执行。

backend 还假设 `ir.launch` 已由 LayoutLoweringPass 创建；如果直接对 Frontend 结果调用 backend，会因为缺少该属性而失败。

## Compiler API

```python
from compiler import compile_kernel

ir, ptx = compile_kernel(src)
print(ir.dump())
print(ir.launch)
print(ptx)
```

`compile_kernel()` 只返回内存中的 IR 和 backend 文本，不创建 output 目录，也不写文件。

## 运行

本版本只依赖 Python 标准库，不要求 GPU：

```bash
cd mini-triton/mini-triton-v0.3/mini-triton-v0.3.1
python demo.py
```

或者使用仓库当前 Conda 环境：

```bash
conda run -n main python demo.py
```

demo 会打印 IR、launch config 和注释形式的 backend 输出，不会初始化 CUDA 或验证数值结果。

## 当前限制

- Tensor shape、dtype、launch 和 mapping 全部固定为 256/FP32/128×2。
- BlockedLayout 已定义但没有被 compiler pipeline 使用。
- 所有 binary operation 都被当成 tensor add。
- 没有 program id、grid、N、mask、广播、归约或多维 tensor。
- LayoutLoweringPass 只追加 metadata/op，没有重写 tensor operations。
- thread_id/lane_id 在 store 后追加，且没有参与实际索引。
- `%tid`、`%lane` 不是正规 SSA Value。
- 没有 layout/launch 一致性或越界验证。
- Backend 只输出注释，不生成可执行 PTX。
- 没有 runtime、单元测试、GPU correctness test 或 benchmark。

## 下一步建议

1. 把 launch、mapping 定义为 FunctionIR 的正式字段。
2. 用 SSA Value 表示 thread_id/lane_id，并在使用点之前生成。
3. 将 tensor operations 重写为每线程负责的标量/向量 operations。
4. 验证 `shape == threads × elements_per_thread` 和 `threads == warps × 32`。
5. 增加 program id、mask 和多 block 覆盖语义。
6. 为 pointer 参数生成 PTX ABI，lower thread id、地址和 load/add/store。
7. 接入 CUDA Driver runtime，并建立正确性测试。
