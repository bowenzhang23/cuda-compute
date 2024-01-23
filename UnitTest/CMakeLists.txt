cmake_minimum_required(VERSION 3.20)

project(gpu_computing_kernal_unittest LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(GTest REQUIRED)

set(target gpu_computing_kernal_unittest)
set(core_name gpu_computing_core)
set(gtest_libraries gtest gtest_main)

include_directories(include/)
include_directories(../Core/include)
include_directories(GoogleTest)

add_executable(${target}.vector test/test_vector.cu)
target_link_libraries(${target}.vector PRIVATE ${core_name} ${gtest_libraries})

add_executable(${target}.matrix test/test_matrix.cu)
target_link_libraries(${target}.matrix PRIVATE ${core_name} ${gtest_libraries})

gtest_discover_tests(${target}.vector)
gtest_discover_tests(${target}.matrix)
install(TARGETS ${target}.vector RUNTIME DESTINATION bin)
install(TARGETS ${target}.matrix RUNTIME DESTINATION bin)
