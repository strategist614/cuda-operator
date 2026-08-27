# NCU 核心诊断树：可复现实例

这里的 CUDA 微基准不是“正确优化写法”，而是故意制造单一、明显的性能特征，帮助把 NCU 指标和代码根因对应起来。

```text
                         Kernel 慢
                            │
                            ▼
                SM Throughput / Memory
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
        SM 高            Memory 高         两个都低
          │                 │                 │
          ▼                 ▼                 ▼
     算力在忙什么？       哪层 Memory？      为什么没指令？
  Tensor / FP32 / 指令     DRAM / L2         Active Warps
                                              │
                                    ┌─────────┴─────────┐
                                    │                   │
                                 Active 低          Active 不少
                                    │                   │
                           Occupancy / Launch           Eligible
                              │          │                │
                          Register     Shared            Stall
                                                   Long/Short Scoreboard
                                                        Barrier
```

## 编译与运行

默认生成 `sm_75` 代码；请按本机 GPU 修改 `ARCH`，例如 Ampere 用 `sm_80` 或 `sm_86`。

```bash
cd profile/ncu-diagnostic-tree
make ARCH=sm_75
./ncu_diagnostic_tree --list
./ncu_diagnostic_tree dram_bandwidth
```

采集一个案例的完整 NCU 报告：

```bash
chmod +x profile_one.sh
./profile_one.sh dram_bandwidth
ncu-ui reports/dram_bandwidth.ncu-rep
```

批量采集全部 10 个案例（每种性能特征生成一个独立报告）：

```bash
chmod +x profile_all.sh
./profile_all.sh
```

如果系统只允许管理员读取 GPU Performance Counters：

```bash
sudo env PATH="$PATH" ./profile_all.sh
sudo chown -R "$USER":"$(id -gn)" reports
```

脚本使用 `--set full`，并让 runner 只启动一次目标 kernel，便于教学。完整集合会进行多次 replay，开销较大。实际迭代时可先用：

```bash
ncu --set basic ./ncu_diagnostic_tree dram_bandwidth --profile-once
```

## 案例与诊断分支

| 案例 | 人为制造的根因 | 预期观察重点 | 对应代码方向 |
|---|---|---|---|
| `tensor_core` | WMMA 循环持续发射 MMA | SM 高，Tensor Core 指令/管线忙 | 检查目标计算单元是否真的被使用 |
| `fp32_pipe` | 8 条独立 FP32 FMA 链 | SM 高，FP32 pipe 忙，Math Pipe Throttle 可能高 | 这是“目标管线吃满”的健康特征 |
| `instructions` | 大量整数、移位和逻辑指令 | SM/issue 活跃，但 Tensor/FP32 不是主因 | 减少索引、地址计算和非目标指令 |
| `dram_bandwidth` | 大数组只读写一次，几乎没有复用 | Memory、DRAM 吞吐高，L2 hit 偏低 | 合并访问、减少字节数、增加复用 |
| `l2_bandwidth` | 反复读取较小工作集 | Memory/L2 活跃，DRAM 相对低，L2 hit 高 | 不要误判成 DRAM 带宽瓶颈 |
| `register_limited` | 每线程保留 128 个独立累加器 | Registers/Thread 高，理论/实际 Occupancy 下降 | 缩小 register tile，并检查 spill |
| `shared_limited` | 每 block 动态申请 48 KiB shared memory | Shared/Block 高，resident blocks 与 Active Warps 低 | 缩小 tile、复用 shared、考虑流水分级 |
| `long_scoreboard` | 大数组上的串行随机 pointer chasing | Active Warps 尚可、Eligible 低、Long Scoreboard 高 | 预取、并发 load、提高复用、重排数据 |
| `short_scoreboard` | 紧密的 shared-memory 地址依赖链 | Short Scoreboard / shared dependency 高 | 增加 ILP、register tiling、重排依赖 |
| `barrier` | 只有一个 warp 做额外工作，其余 warp 等同步 | Barrier stall 高，Eligible 低 | 均衡 warp 工作、减少同步、改用流水 |

`Not Selected` 也常在 `fp32_pipe` 中出现：大量 warp 已 ready，调度器只能选择其中一部分。这通常表示可发射 warp 充足，不应仅凭它数值高就判定为问题。

## 在 NCU 中按树读取

1. 在 **GPU Speed Of Light** 先看 `SM Throughput` 与 `Memory Throughput`。
2. SM 高时打开 **Compute Workload Analysis**，确认是 Tensor、FP32 还是指令发射在忙。
3. Memory 高时打开 **Memory Workload Analysis**，区分 DRAM 吞吐和 L2 命中/吞吐。
4. 两者都低时先看 **Scheduler Statistics** 的 Active/Eligible Warps。
5. Active 低时看 **Occupancy** 与 **Launch Statistics**，再追 Registers/Thread、Shared/Block 和 grid 是否太小。
6. Active 不少但 Eligible 低时，最后才进入 **Warp State Statistics** 看 Stall Reasons。

常用原始指标名（不同 NCU/架构可能略有变化）包括：

```text
sm__throughput.avg.pct_of_peak_sustained_elapsed
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed
lts__t_sector_hit_rate.pct
sm__warps_active.avg.pct_of_peak_sustained_active
smsp__warps_eligible.avg.per_cycle_active
smsp__warps_active.avg.pct_of_peak_sustained_active
```

Stall 指标建议直接在 **Warp State Statistics** 表中比较，而不是跨 kernel 比绝对 cycle 数。

## 推荐实验顺序

先依次对比以下三组报告：

```bash
./profile_one.sh fp32_pipe
./profile_one.sh dram_bandwidth
./profile_one.sh long_scoreboard
```

它们分别对应“算力忙”“DRAM 忙”“SM 和 Memory 都未必吃满、warp 在等长延迟依赖”。随后比较：

```bash
./profile_one.sh register_limited
./profile_one.sh shared_limited
./profile_one.sh barrier
```

每次只写一条因果链：`指标组合 → 假设 → 代码根因 → 准备修改`。不要因为单个 stall 或 occupancy 数字高/低就直接下结论。

## 结果为何不会完全相同

- L2 容量、SM 数量、shared-memory 上限和 Tensor Core 代际都会改变百分比。
- DVFS、其他进程和首次运行会影响耗时，所以普通运行默认 warm-up 后取 3 次平均。
- `l2_bandwidth` 的 4 MiB 工作集若放不进目标 GPU 的 L2，请把源码中的 `n` 调小。
- 编译日志中的 `ptxas info` 可用于先确认寄存器数量；若 `register_limited` 出现 local-memory spill，应减少 `r[128]` 的长度后重测。
- NCU 的 replay 会改变 cache 状态。教学时看指标组合与相对关系，不照抄本文中的“高/低”作为固定阈值。
