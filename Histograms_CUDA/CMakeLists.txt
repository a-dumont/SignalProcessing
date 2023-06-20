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

# Sources
file(GLOB SRC_FILES "src/*.cpp")
file(GLOB CUDA_SRC_FILES "src/*.cu")

# Compiler options
if(CYGWIN)
    add_compile_options(-Wall -O3 -fopenmp -fopenmp-simd -march=native -MMD -MP-DMS_WIN64 -D_hypot=hypot)
elseif(UNIX)
    add_compile_options(-Wall -O3 -march=native -MMD -MP -fopenmp -fopenmp-simd)
endif()

# Target library
if(CYGWIN)
	add_library(histogramscuda ${SRC_FILES})
	SET(OBJS ${CMAKE_CURRENT_SOURCE_DIR}/libs/obj.o)
	ADD_EXECUTABLE(histogramscuda ${OBJS} <other-sources>)
	SET_SOURCE_FILES_PROPERTIES(${OBJS} PROPERTIES EXTERNAL_OBJECT true GENERATED true)
elseif(MSVC)
	add_library(histogramscuda ${SRC_FILES} ${CUDA_SRC_FILES})
	set_target_properties(histogramscuda PROPERTIES CUDA_ARCHITECTURES 60)
elseif(UNIX)
	add_library(histogramscuda ${SRC_FILES} ${CUDA_SRC_FILES})
	set_target_properties(histogramscuda PROPERTIES CUDA_ARCHITECTURES 60)
endif()

# Dependencies
find_package(Python COMPONENTS Interpreter REQUIRED)
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Cygwin stuff
if(CYGWIN)
	# Renaming CUDA toolkit directory
	string(REGEX REPLACE "C\:" "/c" CUDAToolkit_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
    
	# Force naming scheme
	set_target_properties(histogramscuda PROPERTIES SUFFIX ".pyd")
    set_target_properties(histogramscuda PROPERTIES PREFIX "lib")
	
	# Set python executable
	set(Python_EXECUTABLE "/c/Anaconda2/python")
	
	# Linking 
	target_link_libraries(histogramscuda -L/c/Anaconda2/ -lpython27)
    target_link_libraries(histogramscuda -DMS_WIN64 -D_hypot=hypot)
    find_library(CUDART NAMES cudart HINTS "${CUDAToolkit_INCLUDE_DIRS}/../lib/x64")
	target_link_libraries(histogramscuda ${CUDART})
    find_library(CUDA NAMES cuda HINTS "${CUDAToolkit_INCLUDE_DIRS}/../lib/x64")
	target_link_libraries(histogramscuda ${CUDA})
elseif(MSVC)
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
if(CYGWIN)
string(REGEX REPLACE "C\:" "/c" PY_INCL ${PY_INCL})
endif()
separate_arguments(PY_INCL)
include_directories(${PY_INCL})
