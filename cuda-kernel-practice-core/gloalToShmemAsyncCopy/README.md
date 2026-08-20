# Global to Shared Memory Async Copy

本目录计划练习从 global memory 到 shared memory 的异步流水线，包括 `cuda::memcpy_async` 或 PTX `cp.async`、提交/等待分组以及双缓冲。

这类实现通常要求 Ampere（SM 80）或更新架构；编译时需设置匹配的架构，例如 `-arch=sm_80`。验证时应同时检查结果、同步边界和与普通同步加载相比的吞吐。

当前目录尚无源码，是待实现的练习占位。目录名 `gloalToShmemAsyncCopy` 保留了仓库现有拼写。
