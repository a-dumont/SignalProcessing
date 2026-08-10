cmake_minimum_required(VERSION 3.15)

if(NOT CMAKE_BUILD_TYPE)
		set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif()

# Project name
project(vkCompute VERSION 0.0.1)

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
file(GLOB SRC_FILES_vulkanTools "src/vulkanTools/*.cpp")
file(GLOB SRC_FILES_vulkanComputer "src/vkComputer/*.cpp")

# Target library
add_library(vktools ${SRC_FILES_vulkanTools} ${SRC_FILES_vulkanComputer})

target_link_libraries(vktools -lfftw3_omp -lfftw3 -lfftw3f -lfftw3f_omp -lm)
target_link_libraries(vktools -fopenmp -fopenmp-simd)
target_link_libraries(vktools -lvulkan -ldl -lpthread)

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



#set_target_properties( vulkanTools
#    PROPERTIES
#    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib/vulkanTools"
#    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib/vulkanTools"
#)

# computer
#file(GLOB SRC_FILES_computer "src/vkComputer/*.cpp")
#add_library(vkComputer ${SRC_FILES_computer})
#target_link_libraries(vkComputer -fopenmp -fopenmp-simd)
#set_target_properties( vkComputer
#    PROPERTIES
#	ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib/vkComputer"
#	LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib/vkComputer"
#)

# Main executable
#file(GLOB SRC_FILE "src/main.cpp")
#add_executable(vkCompute ${SRC_FILE})
#set_target_properties( vkCompute
#    PROPERTIES
#	RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
#)

#target_link_libraries(vkCompute -lvulkan -ldl -lpthread -lm -lnoise)
#target_link_libraries(vkCraft ${PROJECT_SOURCE_DIR}/lib/primitives/libprimitives.so)
#target_link_libraries(vkCompute ${PROJECT_SOURCE_DIR}/lib/vulkanTools/libvulkanTools.so)
#target_link_libraries(vkCompute ${PROJECT_SOURCE_DIR}/lib/vkComputer/libvkComputer.so)
