cmake_minimum_required(VERSION 3.21)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif()

# Project name
project(histogramscuda LANGUAGES CXX CUDA VERSION 2.2.0)

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
add_library(histogramscuda ${SRC_FILES} ${CUDA_SRC_FILES})
set_target_properties(histogramscuda PROPERTIES CUDA_ARCHITECTURES 60)

# Dependencies
find_package(Python COMPONENTS Interpreter REQUIRED)
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

if(MSVC)
	# Force naming scheme
    set_target_properties(histogramscuda PROPERTIES SUFFIX ".pyd")
    set_target_properties(histogramscuda PROPERTIES PREFIX "lib")
	
	# Set python executable
	set(Python_EXECUTABLE "C:/Anaconda2/python")
    
	# Linking
	find_library(PYLIB NAMES python27 HINTS "C:/Anaconda2/libs")
	target_link_libraries(histogramscuda ${PYLIB})
    find_library(CUDART NAMES cudart HINTS "${CUDAToolkit_INCLUDE_DIRS}/../lib/x64")
	target_link_libraries(histogramscuda ${CUDART})
    find_library(CUDA NAMES cuda HINTS "${CUDAToolkit_INCLUDE_DIRS}/../lib/x64")
	target_link_libraries(histogramscuda ${CUDA})
elseif(UNIX)
    target_link_libraries(histogramscuda -L/opt/cuda/lib64 -lcudart -lcuda)
    target_link_libraries(histogramscuda -fopenmp -fopenmp-simd)
endif()

# pybind11 includes
execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --includes OUTPUT_VARIABLE PY_INCL)
string(REGEX REPLACE "-I" "" PY_INCL ${PY_INCL})
string(REGEX REPLACE "\n" "" PY_INCL ${PY_INCL})

separate_arguments(PY_INCL)
include_directories(${PY_INCL})
