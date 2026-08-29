# High-Performance GPU Top-K V4

V4 的核心目标是把 V3 从：

```text
one fixed Top-16 kernel
```

升级成：

```text
K-specialized WarpSelect kernel family
+
runtime dispatcher
```

此前实测：

```text
RTX 2080
B=128
N=65536
K=16

V0:
525.86 ms

V1:
4.11993 ms

V2:
4.08528 ms

V3:
0.475832 ms
```

V3 相比 V2：

```text
8.59x
```

并且 exact correctness：

```text
PASS
```

---

# 1. V3 仍然存在的浪费

V3 对所有：

```text
1 <= K <= 16
```

内部都维护：

```text
Top-16
```

即使用户只要求：

```text
K=1
K=2
K=4
K=8
```

V3 仍然使用：

```text
lane 0..15
```

并在 merge 中始终支付：

```text
32-element bitonic merge
=
5 stages
```

这对小 K 不合理。

---

# 2. V4 Specialized Kernel Family

V4 新增：

```text
warpselect_k1_v4
warpselect_k2_v4
warpselect_k4_v4
warpselect_k8_v4
warpselect_k16_v4
```

分别由模板实例化：

```cpp
topk_specialized_v4_kernel<1>
topk_specialized_v4_kernel<2>
topk_specialized_v4_kernel<4>
topk_specialized_v4_kernel<8>
topk_specialized_v4_kernel<16>
```

所以：

```text
K
```

在 kernel 内是：

```text
compile-time constant
```

编译器可以：

```text
constant fold
unroll
delete dead branches
```

---

# 3. Warp queue 只维护真正需要的 K

V3：

```text
K=1
仍然维护 16 candidates

K=4
仍然维护 16 candidates
```

V4：

```text
K=1
lane0 owns Top-1

K=2
lanes0..1 own Top-2

K=4
lanes0..3 own Top-4

K=8
lanes0..7 own Top-8

K=16
lanes0..15 own Top-16
```

所以：

```text
logical candidate state
```

随 K 缩小。

---

# 4. V4 仍然保留 V3 的核心 selection 思路

每个 warp 每轮：

```text
load 32 values
        ↓
warp max threshold test
        ↓
batch cannot improve Top-K?
        ├── yes → reject
        └── no
             ↓
        sort 32-element batch
             ↓
        Batch Top-K
             ↓
        merge Current Top-K
```

因此：

```text
V3 最重要的 threshold pruning
```

完全保留。

---

# 5. K-specialized merge network

这是 V4 最主要的改变。

V3 每次 merge 都按：

```text
16 + 16
=
32 elements
```

处理。

固定需要：

```text
stride:
16
8
4
2
1
```

5 stages。

V4：

## K=1

```text
1 + 1 = 2
```

只需要：

```text
stride 1
```

1 stage。

## K=2

```text
2 + 2 = 4
```

需要：

```text
stride 2
1
```

2 stages。

## K=4

```text
4 + 4 = 8
```

3 stages。

## K=8

```text
8 + 8 = 16
```

4 stages。

## K=16

```text
16 + 16 = 32
```

5 stages。

因此 merge complexity：

```text
O(log(2K))
```

而不是统一固定按 K=16 处理。

---

# 6. 为什么仍然先 sort 32-element batch

global load 仍然是一整个 warp：

```text
32 lanes
→
32 consecutive elements
```

所以 batch 输入天然就是：

```text
32 elements
```

V4 仍然先做：

```text
32-lane bitonic sort
```

然后只拿：

```text
first K
```

这是因为如果某个元素连当前：

```text
Batch Top-K
```

都不是，

batch 内已经有 K 个元素比它更好，

它不可能进入最终 global Top-K。

所以 exactness 不受影响。

---

# 7. Dispatcher

V4 默认：

```bash
--kernel auto
```

规则：

```text
K = 1 / 2 / 4 / 8 / 16
        ↓
V4 specialized kernel

other K <= 16
        ↓
V3 fixed Top-16 fallback
```

例如：

```text
K=8
↓
warpselect_k8_v4
```

但：

```text
K=7
↓
warp_batch_threshold_bitonic_v3
```

这已经是一个真正的小型：

```text
Top-K Runtime Dispatcher
```

---

# 8. 编译

```bash
rm -rf build
mkdir build
cd build

cmake ..
cmake --build . -j

cd ..
```

默认：

```text
SM75 / RTX 2080
```

---

# 9. 当前 K=16 第一条实验

因为你已经有 V3 baseline：

```text
0.475832 ms
```

直接：

```bash
./build/topk_bench \
  128 65536 16 \
  --compare-v3
```

V4 默认 auto 会选择：

```text
auto->warpselect_k16_v4
```

注意：

对于：

```text
K=16
```

V4 和 V3 的 merge 网络深度本来都是 5 stages。

因此 **K=16 不一定有巨大提升**。

V4 真正应该明显体现价值的是：

```text
K=1
K=2
K=4
K=8
```

---

# 10. 最重要的 Sweep

运行：

```bash
python3 scripts/sweep.py
```

重点生成：

```text
K    V3 latency    V4 latency    speedup
1
2
4
8
16
```

对于 large-N：

```text
B=128
N=65536
```

尤其有价值。

---

# 11. 手工比较

## K=1

```bash
./build/topk_bench \
  128 65536 1 \
  --compare-v3
```

## K=2

```bash
./build/topk_bench \
  128 65536 2 \
  --compare-v3
```

## K=4

```bash
./build/topk_bench \
  128 65536 4 \
  --compare-v3
```

## K=8

```bash
./build/topk_bench \
  128 65536 8 \
  --compare-v3
```

## K=16

```bash
./build/topk_bench \
  128 65536 16 \
  --compare-v3
```

---

# 12. 所有历史 kernel 仍保留

## V0

```bash
--kernel naive
```

## V1

```bash
--kernel register
```

## V2

```bash
--kernel warp
```

## V3

```bash
--kernel batch
```

## V4

```bash
--kernel specialized
```

## Runtime dispatch

```bash
--kernel auto
```

---

# 13. Nsight Compute

直接：

```bash
NCU=/usr/local/cuda-13.2/bin/ncu \
./scripts/profile_v3_vs_v4.sh
```

默认：

```text
B=128
N=65536
K=16
```

可以改：

```bash
K=4 \
NCU=/usr/local/cuda-13.2/bin/ncu \
./scripts/profile_v3_vs_v4.sh
```

对于 V4，更建议 profile：

```text
K=1
K=4
K=8
```

因为这些才有明显 specialized merge 差异。

---

# 14. V4 NCU 重点

关注：

```text
Instructions Executed

Registers / Thread

Warp Stall Wait

MIO Throttle

Eligible Warps

Achieved Occupancy
```

V4 希望看到小 K：

```text
merge shuffle instruction count ↓
```

并且 compiler 能利用 compile-time K 删除无用代码。

---

# 15. Shuffle Safety

V3 曾经出现过：

```text
__shfl_sync(FULL_MASK)
```

只由半个 warp 执行，

导致 SM75 undefined behavior / apparent hang。

V4 专门遵守：

> mask 中声明参与的所有 lane 都必须执行对应 shuffle。

对于 specialized merge：

```text
K=4
```

只让：

```text
lanes 0..7
```

参与，

mask 就是：

```text
0x000000ff
```

而不是：

```text
FULL_MASK
```

K=16 才使用：

```text
0xffffffff
```

---

# 16. 项目架构到 V4

```text
High-Performance GPU Top-K

                    Runtime Dispatcher
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
       V1 path          V3 path         V4 family
      Register         Warp Batch      K-specialized
      Top-K            Selection       WarpSelect
                                           |
                              +------------+-----------+
                              |   |   |   |            |
                              K1  K2  K4  K8           K16
```

现在项目开始体现：

```text
K changes
    ↓
optimal implementation changes
```

这和之前 GEMM 项目的：

```text
Shape changes
    ↓
optimal tile/config changes
```

形成非常好的呼应。

---

# 17. 下一步应该看什么

V4 的目标不是保证：

```text
K=16
再次 8x
```

因为 V3 对 K=16 已经非常合适。

V4 真正要验证：

> 针对 K 特化是否能显著改善 small-K workload，并形成真实 dispatcher 的必要性。

如果结果显示：

```text
K=1/2/4
V4 明显优于 V3

K=16
V3≈V4
```

这反而是最理想的实验结果。

因为它直接证明：

```text
one kernel does not fit all K
```

从而自然进入后面的：

```text
shape/K adaptive Top-K library
```
