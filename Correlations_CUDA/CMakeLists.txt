cmake_minimum_required(VERSION 3.18)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif()

# Project name
project(correlationscuda LANGUAGES CXX CUDA VERSION 2.1.0)

# C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

# Set output location
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/obj)

# Sources
file(GLOB SRC_FILES "src/*.cxx")
file(GLOB CUDA_SRC_FILES "src/*.cu")

# Compiler options
if(CYGWIN)
    add_compile_options(-Wall -O3 -march=native -MMD -MP -DMS_WIN64 -D_hypot=hypot)
else()
    add_compile_options(-Wall -O3 -march=native -MMD -MP)
endif(CYGWIN)

# Target library
add_library(correlationscuda ${SRC_FILES} ${CUDA_SRC_FILES})

# CUDA stuff
set_target_properties(correlationscuda PROPERTIES CUDA_ARCHITECTURES 62)

# Cygwin stuff
if(CYGWIN)
    set_target_properties(correlationscuda PROPERTIES SUFFIX ".pyd")
    set_target_properties(correlationscuda PROPERTIES PREFIX "lib")
	set(Python_EXECUTABLE "/c/Anaconda2/python")
	target_link_libraries(correlationscuda -L/c/Anaconda2/ -lpython27)
    target_link_libraries(correlationscuda -DMS_WIN64 -D_hypot=hypot)
endif(CYGWIN)

# Dependencies
find_package(Python COMPONENTS Interpreter REQUIRED)
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# pybind11 includes
execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --includes OUTPUT_VARIABLE PY_INCL)
string(REGEX REPLACE "-I" "" PY_INCL ${PY_INCL})
string(REGEX REPLACE "\n" "" PY_INCL ${PY_INCL})
if(CYGWIN)
string(REGEX REPLACE "C\:" "/c" PY_INCL ${PY_INCL})
endif(CYGWIN)
separate_arguments(PY_INCL)
include_directories(${PY_INCL})

# Compiler options
add_compile_options(-std=c++14 -Wall -O3 -march=native -MMD -MP -fopenmp -fopenmp-simd)

target_link_libraries(correlationscuda -L/opt/cuda/lib64 -lcudart -lcufft -lcuda)
