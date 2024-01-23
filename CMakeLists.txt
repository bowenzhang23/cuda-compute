cmake_minimum_required(VERSION 3.20)

add_compile_options(-std=c++20 -O0 -g -Wextra -Wall -Waddress)
# add_compile_options(-std=c++20 -O3 -Wextra -Wall -Waddress)
set(CMAKE_C_COMPILER gcc-12)
set(CMAKE_CXX_COMPILER g++-12)
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
