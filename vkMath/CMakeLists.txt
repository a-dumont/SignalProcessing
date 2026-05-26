cmake_minimum_required(VERSION 3.15)

if(NOT CMAKE_BUILD_TYPE)
		set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif()

# Project name
project(vkCMath VERSION 0.0.1)

# C++ standars
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Shared lib
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

add_compile_options(-Wall -O3 -march=native -DDEBUG -MMD -MP -fopenmp -fopenmp-simd)

# Set output location
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/obj)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/obj)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/obj)

# Sources
file(GLOB SRC_FILES_vkmath "src/*.cpp")

# Target library
add_library(vkmath ${SRC_FILES_vkmath})

target_link_libraries(vkmath -lfftw3_omp -lfftw3 -lfftw3f -lfftw3f_omp -lm)
target_link_libraries(vkmath -fopenmp -fopenmp-simd)
target_link_libraries(vkmath -lvulkan -ldl -lpthread)

target_link_directories(vkmath PRIVATE ../vkCompute/obj/)
target_link_libraries(vkmath -lvktools)

# Dependencies
find_package(Python COMPONENTS Interpreter REQUIRED)

# pybind11 includes
execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --includes OUTPUT_VARIABLE PY_INCL)
string(REGEX REPLACE "-I" "" PY_INCL ${PY_INCL})
string(REGEX REPLACE "\n" "" PY_INCL ${PY_INCL})
if(CYGWIN)
string(REGEX REPLACE "C\:" "/c" PY_INCL ${PY_INCL})
endif(CYGWIN)

separate_arguments(PY_INCL)
include_directories(${PY_INCL})
