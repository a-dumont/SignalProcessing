cmake_minimum_required(VERSION 3.21)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif()

# Project name
project(mathcuda LANGUAGES CXX CUDA VERSION 2.2.0)

# C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

# Set output location
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/obj)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/obj)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/obj)


# Compiler options
if(MSVC)
	#add_compile_options(/Wall /openmp /arch:AVX2)
elseif(UNIX)
    add_compile_options(-Wall -O3 -march=native -MMD -MP)
endif()

# Sources
file(GLOB SRC_FILES "src/*.cpp")
file(GLOB SRC_CUDA "src/*.cu")

# Target library
add_library(mathcuda ${SRC_FILES} ${SRC_CUDA})
set_target_properties(mathcuda PROPERTIES CUDA_ARCHITECTURES 60)

# Dependencies
find_package(Python COMPONENTS Interpreter REQUIRED)
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

if(MSVC)
    set_target_properties(mathcuda PROPERTIES SUFFIX ".pyd")
    set_target_properties(mathcuda PROPERTIES PREFIX "lib")
	set(Python_EXECUTABLE "C:/anaconda3/python")
    find_library(PYLIB NAMES python310 HINTS "C:/anaconda3/libs")
	target_link_libraries(mathcuda ${PYLIB})
    find_library(CUDART NAMES cudart HINTS "${CUDAToolkit_INCLUDE_DIRS}/../lib/x64")
	target_link_libraries(mathcuda ${CUDART})
    find_library(CUDA NAMES cuda HINTS "${CUDAToolkit_INCLUDE_DIRS}/../lib/x64")
	target_link_libraries(mathcuda ${CUDA})
else()
    target_link_libraries(mathcuda -L/opt/cuda/lib64 -lcudart -lcuda)
endif()

# pybind11 includes
execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --includes OUTPUT_VARIABLE PY_INCL)
string(REGEX REPLACE "-I" "" PY_INCL ${PY_INCL})
string(REGEX REPLACE "\n" "" PY_INCL ${PY_INCL})

separate_arguments(PY_INCL)
include_directories(${PY_INCL})
