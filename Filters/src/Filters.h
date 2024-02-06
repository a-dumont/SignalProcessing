#pragma once
#include <stdio.h>
#include <immintrin.h>
#include <cstring>
#include <numeric>
#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <cstdint>
#include <fftw3.h>

#if defined(__CYGWIN__) || defined(__MINGW64__)
    #define _WIN32_WINNT 0x0602 // Windows 8
    #include <windows.h>
    #include <Processtopologyapi.h>
    #include <processthreadsapi.h>
#endif

#ifdef _WIN64
	std::string wisdom_path = "FFTW_Wisdom";
	std::string wisdom_parallel_path = "FFTW_Parallel_Wisdom";
#else
	std::string wisdom_path = "/etc/FFTW/FFTW_Wisdom";
	std::string wisdom_parallel_path = "/etc/FFTW/FFTW_Parallel_Wisdom";
#endif

#include "Filters.tpp"
