cmake_minimum_required(VERSION 3.20)

set(CMAKE_C_COMPILER gcc-12)
set(CMAKE_CXX_COMPILER g++-12)
set(CMAKE_CUDA_COMPILER nvcc)

add_compile_options(
  -std=c++20
  -Wextra
  -Wall
  -Waddress
  $<$<CONFIG:DEBUG>:-g3>
  $<$<CONFIG:DEBUG>:-O0>
  $<$<CONFIG:RELEASE>:-O3>)

add_compile_definitions($<$<CONFIG:DEBUG>:DEBUG_CONSTRUCTOR>
                        $<$<CONFIG:DEBUG>:DEBUG_PERFORMANCE>)

set(CMAKE_CUDA_FLAGS
    "-arch=sm_86 --expt-relaxed-constexpr --use_fast_math -fmad=true")

enable_testing()

project(
  gpu_computing
  VERSION 0.1
  LANGUAGES CXX CUDA)

add_subdirectory(ext/nanobind)
add_subdirectory(Core)
add_subdirectory(UnitTest)

set(CMAKE_INSTALL_PREFIX
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CUDA_LIBRARY_ARCHITECTURE}-${CMAKE_CUDA_COMPILER_VERSION}
)
