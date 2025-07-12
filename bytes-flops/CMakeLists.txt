cmake_minimum_required(VERSION 3.16)
project(bytes_flops_bench CXX)

include(FetchContent)
FetchContent_Declare(
    kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
)

# Enable CUDA backend
option(Kokkos_ENABLE_CUDA "Enable CUDA backend" ON)
option(Kokkos_ENABLE_OPENMP "Enable OpenMP backend" ON)
option(Kokkos_ENABLE_SERIAL "Enable Serial backend" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

FetchContent_MakeAvailable(kokkos)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(bytes_flops_bench
        sources/energy_benchmark.cpp
        sources/bench_float.cpp
        sources/bench_double.cpp
        sources/bench_int32_t.cpp
        sources/bench_int64_t.cpp
)

target_link_libraries(bytes_flops_bench kokkos)

target_compile_options(bytes_flops_bench PRIVATE -O3)
