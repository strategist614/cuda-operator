cmake_minimum_required(VERSION 3.20)

project(
    gpu_topk_v2
    LANGUAGES
    CXX
    CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(
    topk_bench
    src/main.cu
    src/topk_naive.cu
    src/topk_register.cu
    src/topk_warp.cu
)

target_include_directories(
    topk_bench
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    topk_bench
    PRIVATE
    CUDA::cudart
)

target_compile_options(
    topk_bench
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(
    topk_bench
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
