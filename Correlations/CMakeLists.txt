cmake_minimum_required(VERSION 3.15)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif()

# Project name
project(correlations VERSION 2.1.0)


# C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

# Set output location
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/obj)

# Sources
file(GLOB SRC_FILES "src/*.cpp")

# Compiler options
if(CYGWIN)
    add_compile_options(-Wall -O3 -march=native -MMD -MP -fopenmp -fopenmp-simd -DMS_WIN64 -D_hypot=hypot)
else()
    add_compile_options(-Wall -O3 -march=native -MMD -MP -fopenmp -fopenmp-simd)
endif(CYGWIN)

# Target library
add_library(correlations ${SRC_FILES})

# Cygwin stuff
if(CYGWIN)
    set_target_properties(correlations PROPERTIES SUFFIX ".pyd")
    set_target_properties(correlations PROPERTIES PREFIX "lib")
	set(Python_EXECUTABLE "/c/Anaconda2/python")
	target_link_libraries(correlations -L/c/Anaconda2/ -lpython27)
    target_link_libraries(correlations -DMS_WIN64 -D_hypot=hypot)
endif(CYGWIN)

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

target_link_libraries(correlations -lfftw3_omp -lfftw3 -lm)
target_link_libraries(correlations -fopenmp -fopenmp-simd)
