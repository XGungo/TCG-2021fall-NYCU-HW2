find_package(PythonInterp REQUIRED)
cmake_minimum_required(VERSION 3.18)
project(2584)

set(CMAKE_CXX_STANDARD 20)

SET(CMAKE_C_FLAGS_RELEASE "-ffast-math ${CMAKE_C_FLAGS_RELEASE}")
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})
add_executable(2584 2584.cpp)


find_package(Torch REQUIRED)
target_link_libraries(2584 "${TORCH_LIBRARIES}")
set_property(TARGET 2584 PROPERTY CXX_STANDARD 20)

