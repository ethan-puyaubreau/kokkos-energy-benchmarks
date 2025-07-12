cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_saxpy CUDA)

# Set CUDA architectures if needed
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)

# Enable C++11 (or later if needed)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add the executable
add_executable(saxpy saxpy.cu)

# Set CUDA specific flags if needed
set_target_properties(saxpy PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
