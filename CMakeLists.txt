cmake_minimum_required(VERSION 3.18)

project(SignalProcessing VERSION 4.0.0 DESCRIPTION "C++ and CUDA routines wrapped with pybind11")

# Cygwin stuff
if(CYGWIN)
	set(Python_EXECUTABLE "/c/anaconda3/python")
elseif(MSVC)
	set(Python_EXECUTABLE "C:/anaconda3/python")
endif()

# Python
find_package(Python COMPONENTS Interpreter REQUIRED)

message(STATUS "Building the base modules")

add_subdirectory(FFT)
add_subdirectory(Correlations)
add_subdirectory(Histograms)
add_subdirectory(Math)
add_subdirectory(Filters)

if($ENV{ENABLE_CUDA})
		message(STATUS "Building the CUDA modules")
		add_subdirectory(FFT_CUDA)
		add_subdirectory(Correlations_CUDA)
		add_subdirectory(Math_CUDA)
		add_subdirectory(Histograms_CUDA)
endif($ENV{ENABLE_CUDA})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/FFT/obj/
		DESTINATION ${PROJECT_SOURCE_DIR}/SignalProcessing/fft
		FILES_MATCHING PATTERN "libfft.*")
install(DIRECTORY ${PROJECT_SOURCE_DIR}/Correlations/obj/
		DESTINATION ${PROJECT_SOURCE_DIR}/SignalProcessing/correlations
		FILES_MATCHING PATTERN "libcorrelations.*")
install(DIRECTORY ${PROJECT_SOURCE_DIR}/Histograms/obj/
		DESTINATION ${PROJECT_SOURCE_DIR}/SignalProcessing/histograms
		FILES_MATCHING PATTERN "libhistograms.*")
install(DIRECTORY ${PROJECT_SOURCE_DIR}/Math/obj/
		DESTINATION ${PROJECT_SOURCE_DIR}/SignalProcessing/math
		FILES_MATCHING PATTERN "libmath.*")
install(DIRECTORY ${PROJECT_SOURCE_DIR}/Filters/obj/
		DESTINATION ${PROJECT_SOURCE_DIR}/SignalProcessing/filters
		FILES_MATCHING PATTERN "libfilters.*")

if($ENV{ENABLE_CUDA})
	install(DIRECTORY ${PROJECT_SOURCE_DIR}/FFT_CUDA/obj/
		DESTINATION ${PROJECT_SOURCE_DIR}/SignalProcessing/fft
		FILES_MATCHING PATTERN "libfftcuda.*")
	install(DIRECTORY ${PROJECT_SOURCE_DIR}/Correlations_CUDA/obj/
		DESTINATION ${PROJECT_SOURCE_DIR}/SignalProcessing/correlations
		FILES_MATCHING PATTERN "libcorrelationscuda.*")
	install(DIRECTORY ${PROJECT_SOURCE_DIR}/Math_CUDA/obj/
		DESTINATION ${PROJECT_SOURCE_DIR}/SignalProcessing/math
		FILES_MATCHING PATTERN "libmathcuda.*")
	install(DIRECTORY ${PROJECT_SOURCE_DIR}/Histograms_CUDA/obj/
		DESTINATION ${PROJECT_SOURCE_DIR}/SignalProcessing/histograms
		FILES_MATCHING PATTERN "libhistogramscuda.*")
endif($ENV{ENABLE_CUDA})
