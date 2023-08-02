#pragma once
#include <stdio.h>
#include <immintrin.h>
#include <cstring>
#include <numeric>
#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <cstdint>

#if defined(__CYGWIN__) || defined(__MINGW64__)
    #define _WIN32_WINNT 0x0602 // Windows 8
    #include <windows.h>
    #include <Processtopologyapi.h>
    #include <processthreadsapi.h>
#endif

#include "Filters.tpp"
