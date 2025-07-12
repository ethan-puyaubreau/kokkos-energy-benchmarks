cmake_minimum_required(VERSION 3.16)
project(compute_memory_peak CXX)

find_package(Kokkos REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(compute_memory_peak main.cpp)

target_link_libraries(compute_memory_peak Kokkos::kokkos)

target_compile_options(compute_memory_peak PRIVATE -O3)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(compute_memory_peak OpenMP::OpenMP_CXX)
endif()