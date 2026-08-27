#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t error__ = (call);                                                \
    if (error__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(error__));                                \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

namespace wmma = nvcuda::wmma;

constexpr int kBlock = 256;

__global__ void fp32_pipe_kernel(float *output, int inner_iterations) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float x0 = 0.001f * (tid + 1), x1 = x0 + 0.1f;
  float x2 = x0 + 0.2f, x3 = x0 + 0.3f;
  float x4 = x0 + 0.4f, x5 = x0 + 0.5f;
  float x6 = x0 + 0.6f, x7 = x0 + 0.7f;
#pragma unroll 4
  for (int i = 0; i < inner_iterations; ++i) {
    x0 = fmaf(x0, 1.000001f, 0.000001f);
    x1 = fmaf(x1, 0.999999f, 0.000002f);
    x2 = fmaf(x2, 1.000002f, 0.000003f);
    x3 = fmaf(x3, 0.999998f, 0.000004f);
    x4 = fmaf(x4, 1.000003f, 0.000005f);
    x5 = fmaf(x5, 0.999997f, 0.000006f);
    x6 = fmaf(x6, 1.000004f, 0.000007f);
    x7 = fmaf(x7, 0.999996f, 0.000008f);
  }
  output[tid] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}

__global__ void instruction_kernel(std::uint32_t *output,
                                   int inner_iterations) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::uint32_t a = tid + 1u, b = tid ^ 0x9e3779b9u;
  std::uint32_t c = tid + 0x85ebca6bu, d = tid ^ 0xc2b2ae35u;
#pragma unroll 4
  for (int i = 0; i < inner_iterations; ++i) {
    a = (a << 5) | (a >> 27); a += b;
    b ^= (b << 13);           b += c;
    c = (c << 17) | (c >> 15); c ^= d;
    d += 0x9e3779b9u;         d ^= a;
  }
  output[tid] = a ^ b ^ c ^ d;
}

__global__ void dram_bandwidth_kernel(const float *__restrict__ x,
                                      const float *__restrict__ y,
                                      float *__restrict__ output, int n) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) output[tid] = 1.25f * x[tid] + y[tid];
}

__global__ void l2_bandwidth_kernel(const volatile float *input,
                                    float *output, int n,
                                    int inner_iterations) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  float sum = 0.0f;
  // n is a power of two. Every iteration walks the same L2-sized working set.
  for (int i = 0; i < inner_iterations; ++i) {
    const int index = (tid + i * 131) & (n - 1);
    sum += input[index];
  }
  output[tid] = sum;
}

__global__ void __launch_bounds__(128)
register_limited_kernel(float *output, int inner_iterations) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float r[128];
#pragma unroll
  for (int j = 0; j < 128; ++j) r[j] = 0.0001f * (tid + j + 1);

  for (int i = 0; i < inner_iterations; ++i) {
#pragma unroll
    for (int j = 0; j < 128; ++j)
      r[j] = fmaf(r[j], 1.000001f + j * 1.0e-8f, 1.0e-7f);
  }
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < 128; ++j) sum += r[j];
  output[tid] = sum;
}

__global__ void shared_limited_kernel(float *output, int inner_iterations) {
  extern __shared__ float shared[];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float x = 0.001f * (tid + 1);
  shared[threadIdx.x] = x;
  __syncthreads();
  for (int i = 0; i < inner_iterations; ++i)
    x = fmaf(x, 1.000001f, shared[(threadIdx.x + i) & 127] * 1.0e-7f);
  output[tid] = x;
}

__global__ void long_scoreboard_kernel(const std::uint32_t *links,
                                       std::uint32_t *output, int mask,
                                       int inner_iterations) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::uint32_t index = (tid * 2654435761u) & mask;
  // Each load address depends on the preceding, cache-unfriendly load.
  for (int i = 0; i < inner_iterations; ++i) index = links[index];
  output[tid] = index;
}

__global__ void short_scoreboard_kernel(float *output, int inner_iterations) {
  __shared__ volatile float shared[kBlock];
  const int lane = threadIdx.x;
  const int tid = blockIdx.x * blockDim.x + lane;
  shared[lane] = 0.001f * (tid + 1);
  __syncthreads();
  float x = shared[lane];
  // A tight shared-memory read-after-read address dependency.
  for (int i = 0; i < inner_iterations; ++i) {
    const int next = (lane + (__float_as_uint(x) & 31u) + 1) & (kBlock - 1);
    x = shared[next];
  }
  output[tid] = x;
}

__global__ void barrier_kernel(float *output, int outer_iterations) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float x = 0.001f * (tid + 1);
  for (int outer = 0; outer < outer_iterations; ++outer) {
    // Warp 0 does extra work while all other warps soon reach the barrier.
    if (threadIdx.x < 32) {
#pragma unroll 4
      for (int i = 0; i < 128; ++i) x = fmaf(x, 1.000001f, 0.000001f);
    }
    __syncthreads();
    x += 1.0e-7f;
  }
  output[tid] = x;
}

__global__ void tensor_core_kernel(const half *a, const half *b, float *c,
                                   int inner_iterations) {
#if __CUDA_ARCH__ >= 700
  const int warp_in_block = threadIdx.x / warpSize;
  const int global_warp = blockIdx.x * (blockDim.x / warpSize) + warp_in_block;
  const int matrix_offset = global_warp * 16 * 16;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
  wmma::load_matrix_sync(a_frag, a + matrix_offset, 16);
  wmma::load_matrix_sync(b_frag, b + matrix_offset, 16);
  wmma::fill_fragment(c_frag, 0.0f);
  for (int i = 0; i < inner_iterations; ++i)
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  wmma::store_matrix_sync(c + matrix_offset, c_frag, 16, wmma::mem_row_major);
#endif
}

template <class Launch>
float measure(Launch launch, int launches, bool warmup) {
  if (warmup) {
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventRecord(begin));
  for (int i = 0; i < launches; ++i) launch();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, begin, end));
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));
  return elapsed_ms / launches;
}

template <class T>
T *device_alloc(std::size_t count) {
  T *pointer = nullptr;
  CUDA_CHECK(cudaMalloc(&pointer, count * sizeof(T)));
  return pointer;
}

void print_cases() {
  std::puts("tensor_core       SM high: Tensor Core busy");
  std::puts("fp32_pipe         SM high: FP32 math pipe busy");
  std::puts("instructions      SM high: instruction/ALU pressure");
  std::puts("dram_bandwidth    Memory high: DRAM bandwidth");
  std::puts("l2_bandwidth      Memory high: L2 traffic with high reuse");
  std::puts("register_limited  Active warps low: registers limit occupancy");
  std::puts("shared_limited    Active warps low: shared memory limits occupancy");
  std::puts("long_scoreboard   Active warps present, eligible low: global-load latency");
  std::puts("short_scoreboard  Active warps present, eligible low: shared dependency");
  std::puts("barrier           Active warps present, eligible low: barrier imbalance");
}

int main(int argc, char **argv) {
  if (argc < 2 || std::strcmp(argv[1], "--help") == 0) {
    std::fprintf(stderr,
                 "usage: %s CASE [--profile-once] [--launches N]\n"
                 "       %s --list\n", argv[0], argv[0]);
    return argc < 2 ? EXIT_FAILURE : EXIT_SUCCESS;
  }
  if (std::strcmp(argv[1], "--list") == 0) {
    print_cases();
    return EXIT_SUCCESS;
  }

  const std::string which = argv[1];
  int launches = 3;
  bool warmup = true;
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--profile-once") == 0) {
      launches = 1;
      warmup = false;
    } else if (std::strcmp(argv[i], "--launches") == 0 && i + 1 < argc) {
      launches = std::max(1, std::atoi(argv[++i]));
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
      return EXIT_FAILURE;
    }
  }

  float milliseconds = 0.0f;
  if (which == "fp32_pipe") {
    constexpr int blocks = 2048;
    float *out = device_alloc<float>(blocks * kBlock);
    milliseconds = measure([&] { fp32_pipe_kernel<<<blocks, kBlock>>>(out, 4096); },
                           launches, warmup);
    CUDA_CHECK(cudaFree(out));
  } else if (which == "instructions") {
    constexpr int blocks = 2048;
    auto *out = device_alloc<std::uint32_t>(blocks * kBlock);
    milliseconds = measure([&] { instruction_kernel<<<blocks, kBlock>>>(out, 8192); },
                           launches, warmup);
    CUDA_CHECK(cudaFree(out));
  } else if (which == "dram_bandwidth") {
    constexpr int n = 1 << 24;
    float *x = device_alloc<float>(n), *y = device_alloc<float>(n);
    float *out = device_alloc<float>(n);
    CUDA_CHECK(cudaMemset(x, 1, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(y, 2, n * sizeof(float)));
    const int blocks = (n + kBlock - 1) / kBlock;
    milliseconds = measure([&] { dram_bandwidth_kernel<<<blocks, kBlock>>>(x, y, out, n); },
                           launches, warmup);
    CUDA_CHECK(cudaFree(x)); CUDA_CHECK(cudaFree(y)); CUDA_CHECK(cudaFree(out));
  } else if (which == "l2_bandwidth") {
    constexpr int n = 1 << 20;
    float *in = device_alloc<float>(n), *out = device_alloc<float>(n);
    CUDA_CHECK(cudaMemset(in, 1, n * sizeof(float)));
    const int blocks = (n + kBlock - 1) / kBlock;
    milliseconds = measure([&] { l2_bandwidth_kernel<<<blocks, kBlock>>>(in, out, n, 512); },
                           launches, warmup);
    CUDA_CHECK(cudaFree(in)); CUDA_CHECK(cudaFree(out));
  } else if (which == "register_limited") {
    constexpr int threads = 128, blocks = 2048;
    float *out = device_alloc<float>(threads * blocks);
    milliseconds = measure([&] { register_limited_kernel<<<blocks, threads>>>(out, 128); },
                           launches, warmup);
    CUDA_CHECK(cudaFree(out));
  } else if (which == "shared_limited") {
    constexpr int threads = 128, blocks = 2048, shared_bytes = 48 * 1024;
    float *out = device_alloc<float>(threads * blocks);
    milliseconds = measure([&] { shared_limited_kernel<<<blocks, threads, shared_bytes>>>(out, 4096); },
                           launches, warmup);
    CUDA_CHECK(cudaFree(out));
  } else if (which == "long_scoreboard") {
    constexpr int n = 1 << 24, blocks = 512;
    std::vector<std::uint32_t> host(n);
    for (int i = 0; i < n; ++i)
      host[i] = (static_cast<std::uint32_t>(i) * 1664525u + 1013904223u) & (n - 1);
    auto *links = device_alloc<std::uint32_t>(n);
    auto *out = device_alloc<std::uint32_t>(blocks * kBlock);
    CUDA_CHECK(cudaMemcpy(links, host.data(), n * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    milliseconds = measure([&] { long_scoreboard_kernel<<<blocks, kBlock>>>(links, out, n - 1, 512); },
                           launches, warmup);
    CUDA_CHECK(cudaFree(links)); CUDA_CHECK(cudaFree(out));
  } else if (which == "short_scoreboard") {
    constexpr int blocks = 2048;
    float *out = device_alloc<float>(blocks * kBlock);
    milliseconds = measure([&] { short_scoreboard_kernel<<<blocks, kBlock>>>(out, 8192); },
                           launches, warmup);
    CUDA_CHECK(cudaFree(out));
  } else if (which == "barrier") {
    constexpr int blocks = 1024;
    float *out = device_alloc<float>(blocks * kBlock);
    milliseconds = measure([&] { barrier_kernel<<<blocks, kBlock>>>(out, 256); },
                           launches, warmup);
    CUDA_CHECK(cudaFree(out));
  } else if (which == "tensor_core") {
    constexpr int threads = 256, blocks = 1024;
    constexpr int warps = blocks * threads / 32;
    constexpr std::size_t elements = static_cast<std::size_t>(warps) * 16 * 16;
    std::vector<half> host(elements, __float2half(0.01f));
    half *a = device_alloc<half>(elements), *b = device_alloc<half>(elements);
    float *c = device_alloc<float>(elements);
    CUDA_CHECK(cudaMemcpy(a, host.data(), elements * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, host.data(), elements * sizeof(half), cudaMemcpyHostToDevice));
    milliseconds = measure([&] { tensor_core_kernel<<<blocks, threads>>>(a, b, c, 512); },
                           launches, warmup);
    CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b)); CUDA_CHECK(cudaFree(c));
  } else {
    std::fprintf(stderr, "unknown case: %s\n\n", which.c_str());
    print_cases();
    return EXIT_FAILURE;
  }

  std::printf("case=%s average_ms=%.4f launches=%d warmup=%s\n",
              which.c_str(), milliseconds, launches, warmup ? "yes" : "no");
  return EXIT_SUCCESS;
}
