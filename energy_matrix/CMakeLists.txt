cmake_minimum_required(VERSION 3.16)
project(energy_matrix_benchmark)

include(FetchContent)
FetchContent_Declare(
  kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
)

option(Kokkos_ENABLE_CUDA "Enable CUDA backend" ON)
option(Kokkos_ENABLE_OPENMP "Enable OpenMP backend" ON)
option(Kokkos_ENABLE_SERIAL "Enable Serial backend" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

FetchContent_MakeAvailable(kokkos)

find_package(CUDAToolkit QUIET)

set(NVML_FOUND FALSE)

if (NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDAToolkit not found but required for NVML support")
endif()

find_package(CUDA::nvml QUIET)
if(TARGET CUDA::nvml)
    message(STATUS "Found CUDA NVML")
    set(NVML_FOUND TRUE)
else()
    find_library(NVML_LIBRARY NAMES nvidia-ml nvml PATHS ${CUDAToolkit_LIBRARY_DIR})
    if(NVML_LIBRARY)
        message(STATUS "Found NVML: ${NVML_LIBRARY}")
        set(NVML_FOUND TRUE)
    else()
        message(FATAL_ERROR "NVML not found but is required")
    endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(energy_matrix main.cpp)

target_link_libraries(energy_matrix PRIVATE kokkos)

target_include_directories(energy_matrix PRIVATE ../src)

if(NVML_FOUND)
    target_compile_definitions(energy_matrix PRIVATE HAVE_NVML)
    if(TARGET CUDA::nvml)
        target_link_libraries(energy_matrix PRIVATE CUDA::nvml)
    else()
        target_link_libraries(energy_matrix PRIVATE ${NVML_LIBRARY})
    endif()
endif()

target_compile_options(energy_matrix PRIVATE -O3 -march=native)