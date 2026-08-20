# CUDA Kernel Practice Core

本目录集中记录 CUDA 基础 kernel 的实现与优化过程，重点观察线程索引、合并访存、shared memory、warp shuffle 和异步拷贝等机制。

| 子目录 | 内容 | 当前状态 |
| --- | --- | --- |
| `vectorAdd/` | 一维线程索引与向量加 | 可编译示例及 Python benchmark |
| `matrixMul/` | tiled matrix multiplication、pinned memory | CUDA 示例 |
| `reduction/` | 多个 reduction 版本与 warp shuffle | CUDA/C++ 驱动程序 |
| `transpose/` | naive、coalesced、无 bank conflict 等版本 | 多 kernel 对比 |
| `scan/` | Parallel scan | 待实现 |
| `convolutionSeparable/` | 可分离卷积 | 待实现 |
| `gloalToShmemAsyncCopy/` | global-to-shared 异步拷贝 | 待实现；目录名沿用现状 |

各示例没有统一构建脚本。进入含 `.cu` 文件的子目录后使用 `nvcc -O3` 编译，并以该目录 README 的说明为准。
