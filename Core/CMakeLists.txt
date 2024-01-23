cmake_minimum_required(VERSION 3.20)

project(gpu_computing_core LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

find_package(
  Python 3.10
  COMPONENTS Interpreter Development.Module
  REQUIRED)

file(GLOB GPUC_SOURCE "src/*.cpp" "src/*.cu")
file(GLOB GPUC_INCLUDE "include/*.hpp" "include/*.cuh")

set(target gpu_computing_core)

include_directories(include/)

add_library(${target} SHARED ${GPUC_SOURCE} ${GPUC_INCLUDE})
set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS ${target} LIBRARY DESTINATION lib)
install(FILES ${GPUC_INCLUDE} DESTINATION include)

nanobind_add_module(cuda_compute bind/module.cu)
target_link_libraries(cuda_compute PRIVATE ${target})
install(TARGETS cuda_compute LIBRARY DESTINATION lib)
