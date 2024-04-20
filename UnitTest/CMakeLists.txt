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

function(gpuc_add_test target_name test_name)
  set(full_name ${target_name}.${test_name})
  add_executable(${full_name} test/test_${test_name}.cu)
  target_link_libraries(${full_name} PRIVATE ${core_name} ${gtest_libraries})
  gtest_discover_tests(${full_name})
  install(TARGETS ${full_name} RUNTIME DESTINATION bin)
endfunction()

gpuc_add_test(${target} vector)
gpuc_add_test(${target} matrix)
gpuc_add_test(${target} conversion)
