cmake_minimum_required(VERSION 3.21)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif()

# Project name
project(correlationscuda LANGUAGES CXX CUDA VERSION 2.2.0)

# C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

# Set output location
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/obj)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/obj)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/obj)

# Compiler options
if(UNIX)
    add_compile_options(-Wall -O3 -march=native -MMD -MP -fopenmp -fopenmp-simd)
endif()

# Sources
file(GLOB SRC_FILES "src/*.cpp")
file(GLOB CUDA_SRC_FILES "src/*.cu")

# Target library
add_library(correlationscuda ${SRC_FILES} ${CUDA_SRC_FILES})
set_target_properties(correlationscuda PROPERTIES CUDA_ARCHITECTURES 60)

# Dependencies
find_package(Python COMPONENTS Interpreter REQUIRED)
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Cygwin stuff
if(MSVC)
    set_target_properties(correlationscuda PROPERTIES SUFFIX ".pyd")
    set_target_properties(correlationscuda PROPERTIES PREFIX "lib")
	set(Python_EXECUTABLE "C:/anaconda3/python")
    find_library(PYLIB NAMES python310 HINTS "C:/anaconda3/libs")
	target_link_libraries(correlationscuda ${PYLIB})
    find_library(CUFFT NAMES cufft HINTS "${CUDAToolkit_INCLUDE_DIRS}/../lib/x64")
	target_link_libraries(correlationscuda ${CUFFT})
    find_library(CUDART NAMES cudart HINTS "${CUDAToolkit_INCLUDE_DIRS}/../lib/x64")
	target_link_libraries(correlationscuda ${CUDART})
    find_library(CUDA NAMES cuda HINTS "${CUDAToolkit_INCLUDE_DIRS}/../lib/x64")
	target_link_libraries(correlationscuda ${CUDA})
elseif(UNIX)
    target_link_libraries(correlationscuda -L/opt/cuda/lib64 -lcudart -lcufft -lcuda)
endif()

# pybind11 includes
execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --includes OUTPUT_VARIABLE PY_INCL)
string(REGEX REPLACE "-I" "" PY_INCL ${PY_INCL})
string(REGEX REPLACE "\n" "" PY_INCL ${PY_INCL})

separate_arguments(PY_INCL)
include_directories(${PY_INCL})
