# Shape-Adaptive GEMM V7

V7 继续缩小 V6 与 CUTLASS TensorOp mainloop 的差距。

目标环境：

```text
GPU: NVIDIA GeForce RTX 2080
Architecture: Turing / SM75
Input: FP16
Accumulate / Output: FP32
```

V6 已经有：

```text
Global
  ↓
16-byte FP16 vector load
  ↓
Register prefetch
  ↓
2-stage Shared Memory
  ↓
WMMA
  ↓
FP32 accumulator
```

V7 新增第二层 pipeline：

```text
Global → Register → Shared
          pipeline #1

Shared → WMMA operand fragments
          pipeline #2

WMMA → Tensor Core
```

因此 mainloop 从一层流水变成：

```text
CTA pipeline
+
Warp fragment pipeline
```

---

# 1. V7 的三种 Tensor Core path

V7 registry 同时保留三类实现：

## V6 baseline

```text
path = v6_wmma
```

例如：

```text
tcv6_m128_n64_k16_w64x32
```

这是 V6 的原始 winner 结构：

```text
CTA  = 128x64x16
Warp = 64x32
WMMA = 16x16x16
```

---

## V7 fragment pipeline, no padding

```text
path = fragpipe
```

例如：

```text
tcv7_m128_n64_k16_w64x32_frag
```

它保持和 V6 相同的 shared-memory shape，但增加：

```text
Shared → Fragment register double buffering
```

这样可以单独测：

```text
fragment pipeline contribution
```

---

## V7 fragment pipeline + padded shared memory

```text
path = fragpipe_padded
```

例如：

```text
tcv7_m128_n64_k16_w64x32_fp
```

它增加：

```text
fragment double buffer
+
WMMA-compatible shared-memory padding
```

padding：

```text
A stride = BK + 8 half
B stride = BN + 8 half
```

`8 half = 16 bytes`。

这不是 CUTLASS 完整的 XOR TensorOp swizzle。

这里故意保持实现可验证：

```text
V6 regular layout
        ↓
V7 WMMA-compatible padded layout
```

而不是声称已经实现：

```text
CUTLASS XOR permuted layout
+
ldmatrix iterator
```

因为完整的 CUTLASS-style swizzle 需要把 Shared→Register 通路一起改成 native TensorOp operand loading。

---

# 2. Warp fragment double buffering

V6 的一个 BK tile 内大致：

```text
Shared
 ↓
load A fragment
load B fragment
 ↓
mma
 ↓
load next A fragment
load next B fragment
 ↓
mma
```

V7：

```text
load fragment 0
      ↓

load fragment 1
      ↓
mma fragment 0

load fragment 0(next)
      ↓
mma fragment 1
```

代码内部：

```text
a_frag[2][...]
b_frag[2][...]

fragment_stage ^= 1
```

目标：

```text
Shared Memory → Registers latency
        ↓
和
Tensor Core MMA
        ↓
尽可能 overlap
```

---

# 3. 两层 pipeline

V7 的完整 steady state：

```text
Global Tile K+1
      ↓
register prefetch
                       Shared Fragment i+1
                              ↓
Current Shared Tile K         fragment registers
      ↓                              ↓
WMMA Fragment i  → Tensor Core MMA
```

之后：

```text
prefetched Global tile
        ↓
next Shared stage
        ↓
__syncthreads()
        ↓
swap shared stage
```

因此有两个 ping-pong：

```text
Shared stage:
0 ↔ 1

Fragment stage:
0 ↔ 1
```

---

# 4. Shared Memory Padding

V6：

```cpp
half As[2][BM][BK];
half Bs[2][BK][BN];
```

V7 padded path：

```cpp
half As[2][BM][BK + 8];
half Bs[2][BK][BN + 8];
```

为什么不是 `+1`？

WMMA 对 leading dimension 有对齐要求。

FP16 row stride 继续保持：

```text
multiple of 8 half values
```

同时每一行仍保持：

```text
16-byte vector store alignment
```

例如：

```text
BK = 16

V6 A stride:
16 half = 32 B

V7:
24 half = 48 B
```

warp-level fragment 起始位置仍然按 16-row / 16-column tile 对齐。

---

# 5. Native SM75 MMA experimental wrapper

V7 新增：

```text
include/native_mma_sm75.h
```

里面提供：

```text
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
```

的 SM75 inline PTX wrapper。

注意：

**它目前不是默认 GEMM mainloop。**

原因是：

```text
native mma.sync
```

本身不够。

如果想真正替代 WMMA，还需要正确解决：

```text
Shared layout
      ↓
lane-level operand mapping
      ↓
register fragment packing
      ↓
native mma.sync
```

也就是 CUTLASS 的 TensorOp iterator / permuted-layout 那部分。

所以 V7 先把 native MMA 作为下一阶段的明确实验入口，不伪装成已经完成的 native TensorOp GEMM。

---

# 6. 扩大的 Tensor Core Config Space

V7 不再只有 V6 的十来个配置。

当前 registry 包含：

```text
V6 baseline kernels
+
V7 fragment-pipeline kernels
+
V7 padded kernels
+
wider-N CTA kernels
+
BK=64 experiment
```

重点包括：

```text
64x64x16
64x64x32
64x64x64

128x64x16
128x64x32
128x64x48

64x128x16
64x128x32

128x128x16
128x128x32

64x256x16
64x256x32
128x256x16
```

这样可以研究：

```text
reuse
vs
shared memory
vs
register pressure
vs
grid parallelism
```

---

# 7. 编译

```bash
rm -rf build

mkdir build
cd build

cmake ..
cmake --build . -j

cd ..
```

V7 默认仍然针对：

```text
SM75
```

---

# 8. 第一条命令

```bash
./build/shape_gemm \
  128 4096 4096 \
  --retune
```

V7 默认：

```text
dtype=fp16
```

cache：

```text
results/gemm_cache_v7.csv
```

---

# 9. 最重要的 Ablation

你的 V6 winner：

```text
tcv6_m128_n64_k16_w64x32
```

V7 no-padding fragment pipeline：

```text
tcv7_m128_n64_k16_w64x32_frag
```

V7 padded fragment pipeline：

```text
tcv7_m128_n64_k16_w64x32_fp
```

分别运行：

```bash
./build/shape_gemm \
  128 4096 4096 \
  --dtype fp16 \
  --kernel tcv6_m128_n64_k16_w64x32
```

```bash
./build/shape_gemm \
  128 4096 4096 \
  --dtype fp16 \
  --kernel tcv7_m128_n64_k16_w64x32_frag
```

```bash
./build/shape_gemm \
  128 4096 4096 \
  --dtype fp16 \
  --kernel tcv7_m128_n64_k16_w64x32_fp
```

最后得到：

| Implementation | Fragment pipeline | Padding | Latency |
|---|---|---|---:|
| V6 | no | no | ~0.329 ms baseline |
| V7 frag | yes | no | ? |
| V7 frag+pad | yes | yes | ? |

这样可以分开回答：

```text
fragment pipeline 提升多少？
padding 又贡献多少？
```

---

# 10. NCU A/B

直接：

```bash
NCU=/usr/local/cuda-13.2/bin/ncu \
./scripts/profile_v6_vs_v7.sh
```

默认比较：

```text
tcv6_m128_n64_k16_w64x32
```

和：

```text
tcv7_m128_n64_k16_w64x32_fp
```

---

# 11. V7 NCU 重点

## Tensor Core feeding

重点看：

```text
Tensor Core utilization
SM Throughput

Short Scoreboard
MIO Throttle
Wait
Not Selected
```

如果 fragment pipeline 有效，目标是：

```text
Shared → Register operand latency
暴露程度下降
```

---

## Register Pressure

第二层 fragment buffer 不是免费的。

现在 warp 同时保留：

```text
Current A fragments
Current B fragments

+

Next A fragments
Next B fragments

+

Accumulator fragments
```

所以重点看：

```text
Registers / Thread
Achieved Occupancy
Active Warps / SM
```

可能出现：

```text
pipeline latency hiding ↑

但是

register pressure ↑
occupancy ↓
```

最终谁占上风必须实测。

---

## Shared Memory

padding 增加 shared memory：

```text
V6:
2 * (BM*BK + BK*BN) * sizeof(half)

V7 padded:
2 * (
    BM*(BK+8)
    +
    BK*(BN+8)
) * sizeof(half)
```

重点看：

```text
Shared Memory / Block
Shared Load Throughput
Bank Conflicts
Short Scoreboard
```

---

# 12. 你当前 V6 baseline

RTX 2080：

```text
M=128
N=4096
K=4096

tc_m128_n64_k16_w64x32

latency:
0.329021 ms

TFLOPS:
13.0538

cuBLAS Tensor Core:
23.6321 TFLOPS

relative:
55.24%
```

V7 就以这个作为必须超过的 baseline。

---

# 13. 为什么没有直接假装完成 CUTLASS swizzle

CUTLASS SM75 TensorOp 路径的关键不是一个简单：

```text
shared[col ^= something]
```

就结束。

它实际上把：

```text
Global → Shared permutation

Shared → lane operand mapping

native MMA register packing
```

一起设计。

V7 当前仍使用：

```cpp
wmma::load_matrix_sync()
```

所以 shared memory 必须保持 WMMA 能理解的逻辑矩阵 layout。

因此这版做：

```text
WMMA-compatible padding
+
fragment prefetch
```

而不是在 WMMA 背后硬塞一个错误的 XOR layout。

下一次真要做 CUTLASS-style swizzle，就应该连：

```text
native mma.sync
+
operand register mapping
```

一起做。

---

# 14. V7 项目结构

```text
Shape-Adaptive GEMM V7

FP32 SIMT Family
  └── retained from V5

FP16 Tensor Core Family
  │
  ├── V6 WMMA baseline
  │
  ├── V7 fragment pipeline
  │
  └── V7 fragment pipeline + padded SMEM
  │
  ├── CTA double buffer
  ├── Warp fragment double buffer
  ├── 16-byte FP16 global load
  ├── expanded config space
  └── SM75 native MMA experimental wrapper

Kernel Registry
Autotuner
Performance Cache
Runtime Dispatch
Correctness Gate
cuBLAS Tensor Core Baseline
NCU profile mode
```

---

# 15. 最重要的实验结论

V7 不预设：

```text
double-buffer fragment 一定更快
```

因为它可能明显增加 registers。

所以最有价值的结果可能是以下任意一种：

```text
A.
V7 > V6

说明：
Shared→Register feeding 是明显瓶颈。
```

或者：

```text
B.
frag pipeline > V6
但 padded < frag

说明：
pipeline 有用，padding layout 不适合当前 access pattern。
```

或者：

```text
C.
V6 > V7

说明：
V6 已经不是 fragment-load limited，
而 V7 的 register pressure / occupancy cost 更大。
```

三种结果都能指导真正的下一步，而不是凭感觉继续优化。
