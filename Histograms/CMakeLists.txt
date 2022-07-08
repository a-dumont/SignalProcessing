cmake_minimum_required(VERSION 3.15)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif()

# Project name
project(histograms VERSION 2.1.0)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")

# Set output location
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/obj)

# Sources
file(GLOB SRC_FILES "src/*.cpp")

# Target library
add_library(histograms ${SRC_FILES})

# Cygwin stuff
if (CYGWIN)
    set_target_properties(histograms PROPERTIES SUFFIX ".pyd")
	set(Python_EXECUTABLE "/c/Anaconda2/python")
	target_link_options(histograms PUBLIC "LINKER:-L /c/Anaconda2/ -lpython27")
endif (CYGWIN)

# Dependencies
find_package(Python COMPONENTS Interpreter REQUIRED)

# pybind11 includes
execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --includes OUTPUT_VARIABLE PY_INCL)
string(REGEX REPLACE "-I" "" PY_INCL ${PY_INCL})
string(REGEX REPLACE "\n" "" PY_INCL ${PY_INCL})
separate_arguments(PY_INCL)
include_directories(${PY_INCL})

# Compiler options
add_compile_options(-std=c++17 -Wall -O3 -march=native -MMD -MP -fopenmp -fopenmp-simd)

target_link_options(histograms PUBLIC "LINKER:-fopenmp,-fopenmp-simd")