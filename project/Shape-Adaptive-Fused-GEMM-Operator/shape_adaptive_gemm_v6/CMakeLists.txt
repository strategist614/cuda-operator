cmake_minimum_required(VERSION 3.20)

project(
    shape_adaptive_gemm_v6
    LANGUAGES
    CXX
    CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# V6 targets the user's Turing / RTX 2080 environment by default.
# Override with -DCMAKE_CUDA_ARCHITECTURES=XX if needed.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

find_package(
    CUDAToolkit
    REQUIRED
)

add_executable(
    shape_gemm

    src/main.cu

    # FP32 SIMT family (V5)
    src/gemm_kernels.cu
    src/autotuner.cu

    # FP16 Tensor Core family (V6)
    src/tensor_core_kernels.cu
    src/tc_autotuner.cu

    # Unified performance DB
    src/cache.cpp
)

target_include_directories(
    shape_gemm

    PRIVATE

    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    shape_gemm

    PRIVATE

    CUDA::cudart
    CUDA::cublas
)

target_compile_options(
    shape_gemm

    PRIVATE

    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(
    shape_gemm

    PROPERTIES

    CUDA_SEPARABLE_COMPILATION ON
)
