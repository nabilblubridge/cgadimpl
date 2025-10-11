cmake_minimum_required(VERSION 3.20)
project(agkernels_cpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

include_directories(${CGADIMPL_INCLUDE_DIR})

# ===========================================================
# --- 1. Main Shared Library (CPU + CUDA)
# ===========================================================
add_library(agkernels_cpu SHARED
    src/agkernels_cpu.cpp
    src/matker.cu
)

set_target_properties(agkernels_cpu PROPERTIES
    OUTPUT_NAME "agkernels_cpu"
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(agkernels_cpu PRIVATE
  -O3 -mavx2 -mfma -fopenmp
  -Wall -Wextra -Wpedantic
)

target_link_libraries(agkernels_cpu
    PRIVATE
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
)

include(GNUInstallDirs)
install(TARGETS agkernels_cpu LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# ===========================================================
# --- 2. Benchmark Targets (still optional)
# ===========================================================
function(add_matmul_benchmark target_name source_file)
    add_executable(${target_name}
        benchmark/${source_file}
        src/agkernels_cpu.cpp
        src/matker.cu
    )

    target_link_libraries(${target_name}
        PRIVATE Eigen3::Eigen
                OpenMP::OpenMP_CXX
    )

    target_compile_options(${target_name} PRIVATE -O3 -mavx2 -mfma -fopenmp)
    message(STATUS "Added benchmark target: ${target_name}")
endfunction()

add_matmul_benchmark(test_latency      test_matmul_latency.cpp)
add_matmul_benchmark(test_throughput   test_matmul_throughput.cpp)
add_matmul_benchmark(test_aspect       test_matmul_aspect.cpp)
add_matmul_benchmark(test_scalability  test_matmul_scalability.cpp)
add_matmul_benchmark(test_cache        test_matmul_cache.cpp)
