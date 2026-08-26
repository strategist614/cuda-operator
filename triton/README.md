# Triton Kernels

本目录使用 Triton 编写逐元素和行归约 GPU kernel，帮助对照 CUDA 中的 grid、block、线程索引、越界判断及 block 内 reduction。

| 文件 | 状态 |
| --- | --- |
| `add.py` | 完整实现，包含 PyTorch reference 与正确性检查 |
| `relu.py` | 完整实现，包含 PyTorch reference 与正确性检查 |
| `sigmoid.py` | 完整实现，包含 PyTorch reference 与正确性检查 |
| `layernorm.py` | 最后一维 LayerNorm，FP32 统计量、weight/bias 融合与正确性检查 |
| `rmsnorm.py` | 最后一维 RMSNorm，FP32 统计量、weight 融合与正确性检查 |
| `softmax.py` | 数值稳定的最后一维 Softmax 与正确性检查 |
| `normalization_benchmark.py` | 统一比较 LayerNorm、RMSNorm、Softmax 的 Triton 与 PyTorch 性能 |
| `gemm.py` | 固定 tile、单 stage 的普通 Triton GEMM 基线 |
| `gemm_optimized.py` | Tensor Core、grouped ordering、autotune 和多 stage 流水 GEMM |
| `gemm_sm86_compile_check.py` | 离线编译 `sm_86` 并检查 `mma.sync`、`cp.async` 和双缓冲 |
| `gemm_benchmark.py` | 比较普通版、高优化版和 PyTorch GEMM |
| `relu_benchmark.py` | 扫描 ReLU block size，统计寄存器、时间和有效带宽，并导出汇编 |
| `relu_256.ptx` | `BLOCK_SIZE=256` 时导出的 PTX |
| `relu_256.sass` | `BLOCK_SIZE=256` 时导出的 SASS |

## Add

`add.py` 实现两个相同 shape 的连续 CUDA tensor 的逐元素相加：

```text
output[i] = x[i] + y[i]
```

`add_kernel` 通过 `tl.program_id(axis=0)` 获取当前 program 编号，然后生成该 program 负责的 offsets。kernel 分别加载 `x` 和 `y`，完成加法后写入输出。`triton_add()` 检查输入设备、shape 和连续性，使用 `torch.empty_like()` 创建输出，并通过 `triton.cdiv()` 计算 grid。

测试使用 1000 个元素，故意让长度不能被 `BLOCK_SIZE=256` 整除，以验证最后一个 program 的 mask。结果使用 `torch.testing.assert_close` 与 `x + y` 比较。

```bash
python add.py
```

## ReLU

`relu.py` 实现逐元素 ReLU：

```text
y[i] = max(x[i], 0)
```

`relu_kernel` 加载输入后使用 `tl.maximum(x, 0.0)` 完成计算。`triton_relu()` 要求输入位于 CUDA 且内存连续，按 256 个元素一个 program 启动 kernel。测试输入由 `torch.randn()` 生成，因此同时覆盖正数和负数，并使用 `torch.relu(x)` 作为 reference。

```bash
python relu.py
```

### ReLU benchmark

`relu_benchmark.py` 使用 `N = 16 × 1024 × 1024` 个 FP32 元素，对下面几种配置进行测试：

```text
BLOCK_SIZE = 64, 128, 256, 512, 1024
num_warps  = 4
```

每个配置先启动一次 kernel 以触发编译并取得编译结果，再用 `triton.testing.do_bench` 测量重复执行时间。脚本输出：

- `BLOCK`：每个 Triton program 处理的元素数量。
- `regs`：编译后的 kernel 每线程使用的寄存器数量。
- `time`：kernel 执行时间，单位为毫秒。
- `bw`：根据一次读取和一次写入计算的有效显存带宽，单位为 GB/s。

有效带宽的计算方式为：

```text
bytes_moved = 2 × N × sizeof(float)
GB/s = bytes_moved / time_seconds / 1e9
```

这里的系数 2 表示读取 `x` 和写入 `y`。这是便于比较不同配置的算法有效带宽，不等同于 profiler 统计的所有实际 DRAM 流量。

运行 benchmark：

```bash
python relu_benchmark.py
```

脚本还会在 `BLOCK_SIZE=256` 时将编译结果保存为 `relu_256.ptx` 和 `relu_256.sass`，并打印 SASS 中包含 `LDG` 或 `STG` 的 global-memory 指令。生成文件会写入执行命令时的当前工作目录，因此建议在 `triton/` 目录内运行。

> `kernel._init_handles()` 和 `kernel.asm` 属于较底层的编译结果接口，可能随 Triton 版本变化；本仓库当前环境使用 Triton 3.7.1。

## Sigmoid

`sigmoid.py` 实现逐元素 Sigmoid：

```text
y = 1 / (1 + exp(-x))
```

`sigmoid_kernel` 使用 `tl.exp(-x)` 计算指数，然后将结果写回输出。`triton_sigmoid()` 负责创建输出、计算一维 grid 并启动 kernel。测试使用 `torch.sigmoid(x)` 作为 reference，并通过 `torch.testing.assert_close` 检查结果。

当前实现表达式清晰，适合学习 Triton 基础；如果用于极端输入或更严格的数值场景，还应针对大正数、大负数和不同 dtype 单独测试误差。

```bash
python sigmoid.py
```

## LayerNorm、RMSNorm 与 Softmax

`layernorm.py`、`rmsnorm.py` 和 `softmax.py` 都把输入的最后一维视为一行，每个 Triton program 处理一行。其他前导维会展平为 `n_rows`，所以二维和更高维连续 tensor 使用相同的 kernel：

```text
n_cols = x.shape[-1]
n_rows = x.numel() / n_cols
grid   = (n_rows,)
```

`BLOCK_SIZE` 取大于等于 `n_cols` 的最小 2 次幂，超过实际列数的位置通过 mask 保护。因此测试同时覆盖 `hidden=1000` 的非 2 次幂输入和 `hidden=1024` 的对齐输入。

### LayerNorm

LayerNorm 在一个 program 中完成均值、方差、归一化和仿射变换：

```text
mean     = sum(x) / n_cols
variance = sum((x - mean)^2) / n_cols
y        = (x - mean) * rsqrt(variance + epsilon) * weight + bias
```

### RMSNorm

RMSNorm 不减去均值，也不使用 bias：

```text
mean_square = sum(x^2) / n_cols
y           = x * rsqrt(mean_square + epsilon) * weight
```

### Softmax

Softmax 先减去每行最大值，避免直接计算大指数时溢出：

```text
shifted = x - max(x)
y       = exp(shifted) / sum(exp(shifted))
```

三个 kernel 都把输入转换为 FP32 后完成 reduction，再转换回输入 dtype 写入输出。Python wrapper 支持连续的 FP16、BF16 和 FP32 CUDA tensor；BF16 需要 GPU 架构支持。LayerNorm 和 RMSNorm 要求 weight/bias 与输入位于同一设备、使用相同 dtype，并且长度等于最后一维。

当前实现是 forward-only 学习示例，不带自定义 autograd backward。每个 program 需要把整行放入一个 Triton block tensor，因此当前限制最后一维不超过 65536；非常长的行应使用分段 reduction。

运行 PyTorch reference 测试：

```bash
python layernorm.py
python rmsnorm.py
python softmax.py
```

### RTX 2080 正确性结果

测试环境为 RTX 2080、PyTorch 2.12.1+cu132、Triton 3.7.1。RTX 2080 不原生支持 BF16，因此本次实测覆盖 FP32 和 FP16：

| Kernel | Shape | Dtype | 最大绝对误差 | 状态 |
| --- | --- | --- | ---: | --- |
| `triton_layer_norm` | `(257, 1000)` | FP32 | 0.00000191 | PASS |
| `triton_layer_norm` | `(4, 8, 1024)` | FP32 | 0.00000143 | PASS |
| `triton_layer_norm` | `(257, 1000)` | FP16 | 0.00195312 | PASS |
| `triton_layer_norm` | `(4, 8, 1024)` | FP16 | 0.00195312 | PASS |
| `triton_rms_norm` | `(257, 1000)` | FP32 | 0.00000191 | PASS |
| `triton_rms_norm` | `(4, 8, 1024)` | FP32 | 0.00000048 | PASS |
| `triton_rms_norm` | `(257, 1000)` | FP16 | 0.00195312 | PASS |
| `triton_rms_norm` | `(4, 8, 1024)` | FP16 | 0.00000000 | PASS |
| `triton_softmax` | `(257, 1000)` | FP32 | 0.00000012 | PASS |
| `triton_softmax` | `(4, 8, 1024)` | FP32 | 0.00000003 | PASS |
| `triton_softmax` | `(257, 1000)` | FP16 | 0.00000763 | PASS |
| `triton_softmax` | `(4, 8, 1024)` | FP16 | 0.00000000 | PASS |

Softmax 测试还检查了每一行概率和：FP32 最大偏差不超过 `2.4e-7`，FP16 最大偏差不超过 `1.2016e-4`。

### 统一 benchmark

`normalization_benchmark.py` 使用同一组输入依次比较三个 Triton kernel 与 PyTorch 原生实现：

| 算子 | Triton | PyTorch reference |
| --- | --- | --- |
| LayerNorm | `triton_layer_norm` | `torch.nn.functional.layer_norm` |
| RMSNorm | `triton_rms_norm` | `torch.nn.functional.rms_norm` |
| Softmax | `triton_softmax` | `torch.softmax` |

默认测试 `4096 x 1024`、FP32 和 FP16，每个实现 warm-up 25 次、重复计时 100 次。首次 Triton JIT 不计入时间，每组计时前还会调用 `torch.testing.assert_close` 检查结果：

```bash
python normalization_benchmark.py
```

也可以修改 shape、dtype 和计时次数：

```bash
python normalization_benchmark.py \
    --rows 8192 \
    --hidden 2048 \
    --dtype fp16 \
    --warmup 25 \
    --repetitions 200
```

`--dtype all` 在 Turing GPU 上测试 FP32/FP16，在 Ampere 或更新 GPU 上还会加入 BF16。有效带宽统一按最少一次输入读取和一次输出写入计算：

```text
effective_gbps = 2 * x.numel() * x.element_size() / (average_ms * 10^6)
```

它是方便横向比较的算法有效带宽，没有计入 weight/bias 流量、cache 行为或底层实现的额外访存，不等同于 profiler 的 DRAM throughput。

RTX 2080、PyTorch 2.12.1+cu132、Triton 3.7.1 的实测结果：

| 算子 | Dtype | Triton (ms) | PyTorch (ms) | Triton 有效带宽 (GB/s) | PyTorch 有效带宽 (GB/s) | 加速 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LayerNorm | FP32 | 0.08870311 | 0.10143384 | 378.2780 | 330.8012 | 1.1435x |
| RMSNorm | FP32 | 0.08793647 | 0.09653577 | 381.5758 | 347.5855 | 1.0978x |
| Softmax | FP32 | 0.08758260 | 0.09723496 | 383.1176 | 345.0861 | 1.1102x |
| LayerNorm | FP16 | 0.04681269 | 0.05677113 | 358.3903 | 295.5237 | 1.2127x |
| RMSNorm | FP16 | 0.04643134 | 0.04802869 | 361.3339 | 349.3165 | 1.0344x |
| Softmax | FP16 | 0.04624132 | 0.05179807 | 362.8187 | 323.8966 | 1.1202x |

当前 shape 上六组 Triton 实现均快于对应 PyTorch forward。这里的差异只代表指定硬件、版本、shape 和 dtype；修改 hidden size 后，`BLOCK_SIZE`、warp 数、padding 比例和 PyTorch kernel 路径都会变化，应重新 benchmark。

## GEMM

本目录提供普通版和面向 Ampere 的高优化版，均计算连续行主序矩阵：

```text
A: [M, K]
B: [K, N]
C: [M, N]
C = A @ B
```

两个 kernel 都使用 FP32 accumulator，并在写回时转换为输出 tensor 的 dtype。M/N/K 不要求是 tile 的整数倍。

### 普通版 `gemm.py`

普通版使用固定的 `32 x 32 x 32` tile 和二维 program grid：

```text
grid = (ceil(M / 32), ceil(N / 32))
num_warps  = 4
num_stages = 1
```

每个 program 沿 K 维逐块加载 A/B，通过 `tl.dot` 累加，再用 M/N mask 写回。它没有 grouped program ordering、autotune 或多 stage pipeline，支持 FP16 和 FP32，主要作为结构清晰的性能基线。

### 高优化版 `gemm_optimized.py`

高优化版要求 compute capability 8.0 或更高，面向 RTX 3090 的 `sm_86` 路径包含：

- FP16/BF16 输入、FP32 accumulator 的 `tl.dot` Tensor Core 计算。
- 一维 grid 与 `GROUP_SIZE_M=8` 的 grouped tile ordering，连续处理一组 M tiles，提高 B tile 的 L2 cache 复用机会。
- `64/128` 的 M/N tile 和 `BLOCK_K=32` 的多组候选配置。
- `num_warps=4/8`、`num_stages=2/3` autotune，根据 M/N/K 选择实测最快配置。
- K-loop 使用 `tl.range(..., num_stages=PIPELINE_STAGES)`，让后续 A/B tile 的 global-to-shared copy 与当前 tile 的 MMA 重叠。
- 边界 M/N 使用取模加载，避免复杂边界 mask 阻断 dot operand pipeline；最终 C 写回仍使用真实边界 mask。

候选配置中的 `PIPELINE_STAGES` 均不小于 2，因此不会退化为单 stage 版本。`64 x 64 x 32`、3-stage 配置在 `sm_86` 编译结果中实际分配 16 KiB shared memory：单套 A/B tile 为 8 KiB，对应两套 shared buffer。

#### Triton 中的 WMMA

Triton 不直接暴露 CUDA C++ 的 `nvcuda::wmma::fragment` API。Triton 源码使用 `tl.dot` 表达矩阵乘加，NVIDIA backend 再把它降低为 PTX `mma.sync` 和最终的 Tensor Core 机器指令。因此这里所说的 WMMA/Tensor Core 路径是 `tl.dot -> mma.sync`，不是在 Python 中直接调用 WMMA 类型。

当前 Triton 3.7.1 backend 在 `sm_75` 上会把本 kernel 的 `tl.dot` 降低为 SIMT `fma.rn.f32`，并且不启用 Ampere loop pipeline。因此高优化 wrapper 明确拒绝 `sm_75`，避免把 RTX 2080 fallback 错称为 Tensor Core + double buffering；应在 RTX 3090 或更新 GPU 上运行。

### `sm_86` 离线编译检查

即使当前机器不是 Ampere，也可以通过 Triton 的离线编译接口生成 `sm_86` PTX，并检查 Tensor Core 和双缓冲证据：

```bash
python gemm_sm86_compile_check.py
```

本次 CUDA 13.2、Triton 3.7.1 静态编译结果：

| 指标 | 结果 |
| --- | ---: |
| Target | `sm86` |
| Warps | 4 |
| Pipeline stages | 3 |
| Static shared memory | 16,384 bytes |
| `mma.sync` 数量 | 16 |
| `cp.async` 数量 | 20 |
| `cp.async.commit_group` | 存在 |
| `cp.async.wait_group` | 存在 |
| Tensor Core 检查 | PASS |
| Double buffering 检查 | PASS |

其中 PTX 的核心指令包括：

```text
cp.async.cg.shared.global
cp.async.commit_group
cp.async.wait_group
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

`gemm_sm86_compile_check.py` 使用 Triton 的编译器内部接口，适用于仓库记录的 Triton 3.7.1；升级 Triton 后如果内部 API 变化，需要同步调整脚本。

### 正确性测试

普通版已在 RTX 2080 上实测：

| Shape (M x N x K) | Dtype | 最大绝对误差 | 状态 |
| --- | --- | ---: | --- |
| `257 x 259 x 384` | FP32 | 0.00000334 | PASS |
| `257 x 259 x 384` | FP16 | 0.00195312 | PASS |
| `512 x 512 x 512` | FP16 | 0.00195312 | PASS |

优化版修改后的 M/N 取模边界路径也通过了 `257 x 259 x 384` 数学结果检查，最大绝对误差为 `0.00390625`。该检查在 RTX 2080 上只验证计算和边界逻辑；Tensor Core 与 `cp.async` 由上述 `sm_86` 离线编译确认，最终 runtime 正确性与性能仍需在 RTX 3090 上复测。

运行：

```bash
python gemm.py
python gemm_optimized.py
```

在 `sm_75` 上，`gemm_optimized.py` 会输出 `runtime_status=SKIP`；在 `sm_80+` 上会执行 FP16 correctness test、autotune 并打印最优配置。

### GEMM benchmark

```bash
python gemm_benchmark.py
```

可以自定义矩阵尺寸和计时次数：

```bash
python gemm_benchmark.py \
    --m 4096 \
    --n 4096 \
    --k 4096 \
    --warmup 25 \
    --repetitions 100
```

脚本使用 FP16 输入，先和 `torch.matmul` 做正确性检查，再用 `triton.testing.do_bench` 测量时间并按 `2MNK` 计算 GFLOPS。benchmark 的默认矩阵尺寸为 `4096 x 4096 x 4096`；在 `sm_80+` GPU 上会输出普通版、高优化版、PyTorch、最优 autotune 配置以及相对加速比。

RTX 4090 上 `4096 x 4096 x 4096` 的实测结果：

| Provider | 平均时间 (ms) | GFLOPS | 状态 |
| --- | ---: | ---: | --- |
| 普通 Triton GEMM | 1.68614340 | 81510.8334 | PASS |
| 高优化 Triton GEMM | 0.82648356 | 166293.6328 | PASS |
| PyTorch `matmul` | 0.81394030 | 168856.3069 | PASS |

高优化版相比普通版达到 `2.0401x` 加速，执行时间减少约 `50.98%`；吞吐达到 PyTorch 的 `98.48%`，PyTorch 只领先约 `1.54%`。本次 autotune 选择：

```text
BLOCK_M=64, BLOCK_N=128, BLOCK_K=32
GROUP_SIZE_M=8, PIPELINE_STAGES=2
num_warps=4, num_stages=2
```

本次 autotune 选择了较宽的 `64 x 128` 输出 tile 和 2-stage pipeline。`4096³` 提供了足够多的 program 填满 GPU，更大的 N tile 可以增加每个 program 的计算量并减少调度开销；最终配置仍以 autotune 在当前 shape 上的实测为准。

## Conda 环境

本仓库当前使用 Conda 的 `main` 环境：

| 组件 | 版本 |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.12.1+cu132 |
| PyTorch CUDA | 13.2 |
| Triton | 3.7.1 |
| NumPy | 2.2.6 |

激活环境并检查 GPU 是否可用：

```bash
conda activate main
python -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__); print(torch.cuda.is_available())"
```

`torch.cuda.is_available()` 应输出 `True`。如果是 `False`，请检查 NVIDIA 驱动、GPU 访问权限，以及 PyTorch 所带 CUDA runtime 与驱动的兼容性。

## 运行示例

在 `main` 环境中运行：

```bash
conda activate main
cd triton
python add.py
python relu.py
python sigmoid.py
python layernorm.py
python rmsnorm.py
python softmax.py
python normalization_benchmark.py
python gemm.py
python gemm_optimized.py
python gemm_sm86_compile_check.py
python gemm_benchmark.py
```

## 逐元素示例的共同执行模型

三个示例都使用一维 grid，每个 Triton program 处理 `BLOCK_SIZE=256` 个元素：

```text
pid         = tl.program_id(axis=0)
offsets     = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask        = offsets < n_elements
program 数量 = ceil(n_elements / BLOCK_SIZE)
```

mask 用于保护最后一个不完整的数据块，避免越界读写。示例假设输入位于 CUDA 且内存连续；Add 还要求两个输入 shape 相同。`add.py`、`relu.py` 和 `sigmoid.py` 主要用于正确性学习；目前独立的性能测试集中在 `relu_benchmark.py`。

## 为什么这里不需要显式向量化加载

在 CUDA C++ 中，常用 `float4` 让一个线程一次加载或存储 4 个连续的 `float`：

```cpp
float4 value = reinterpret_cast<const float4*>(input)[index];
```

本目录的 Add、ReLU 和 Sigmoid 没有写类似的显式向量类型，因为 Triton 的编程模型不同。下面的 `offsets` 不是单个标量下标，而是一个包含 `BLOCK_SIZE` 个连续下标的 Triton tensor：

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
x = tl.load(x_ptr + offsets, mask=mask)
```

一次 `tl.load` 描述了整个 program 对一批连续元素的加载。编译器可以根据地址连续性、数据类型、对齐和目标 GPU，把这些访问降级为合适的向量化指令与合并的 global-memory transactions。因此，这三个 kernel 已经表达了批量连续访问，没有必要为了模仿 CUDA 再手写 `float4`。

`relu_benchmark.py` 导出的 `relu_256.sass` 可以直接验证这一点。在当前编译环境下，其中的主要 global-memory 指令为：

```text
LDG.E.64.SYS
STG.E.64.SYS
```

`.64` 表示该机器指令按 64 bit（两个 FP32 元素）执行加载和存储。也就是说，Triton 源码虽然没有显式写 `float2`/`float4`，后端仍根据 program 布局生成了向量化访存。具体宽度依赖编译配置、Triton 版本和目标 GPU，不能假设所有环境都会得到完全相同的指令。

对这三个逐元素算子来说，连续 offsets 还带来以下好处：

- 相邻执行单元访问相邻地址，便于 global memory coalescing。
- Add 的两个输入和一个输出都按相同连续下标访问。
- ReLU、Sigmoid 在加载后直接计算并写回，不需要 shared memory 做数据复用。
- mask 可以安全处理元素数量不是 `BLOCK_SIZE` 整数倍的尾部。

不过，不显式使用 `float4` 不代表性能一定自动达到最优。实际优化时仍需 benchmark，并根据 dtype、shape 和 GPU 调整 `BLOCK_SIZE`、`num_warps`；对于二维或非连续数据，还需要设计合适的 block 布局，并可通过 `tl.multiple_of`、`tl.max_contiguous` 等提示向编译器提供已知的对齐或连续性信息。是否生成理想访存指令，应以 Triton 编译结果和 profiler 数据为准。
