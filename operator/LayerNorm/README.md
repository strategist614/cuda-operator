## LayerNorm

### 安装 nsys
```
apt update
apt install -y --no-install-recommends gnupg lsb-release ca-certificates wget software-properties-common

echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" \
  > /etc/apt/sources.list.d/nvidia-devtools.list

apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

apt update
apt install -y nsight-systems-cli
```

```
输入矩阵
4096 rows × 1024 cols

             GPU Grid
────────────────────────────────

Block 0     → 负责 row 0
Block 1     → 负责 row 1
Block 2     → 负责 row 2
...
Block 4095  → 负责 row 4095


每一个 Block:

256 threads
│
├─ thread 0   → col 0,256,512,768
├─ thread 1   → col 1,257,513,769
├─ thread 2   → col 2,258,514,770
│
│      ...
│
└─ thread 255 → col 255,511,767,1023
```

### 优化思路

* 在 `reduction` 那里 每次都需要 `__syncthreads()` 有很多次 `block synchronization`
  所以优化思路是 `warp` 内不需要 `shared memory reduction`
  我现在是 `256` 个 `threads` 一个 `warp` 有 `32` 个 `threads` 所以有 `8` 个 `wraps` 
  `__shfl_down_sync()` 在 `wrap` 中能直接