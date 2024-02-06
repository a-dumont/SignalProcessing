#pragma once
#include <stdio.h>
#include <complex>
#include <numeric>
#include <stdlib.h>
#include <cmath>
#include <omp.h>

#if defined(__CYGWIN__) || defined(__MINGW64__)
    #define _WIN32_WINNT 0x0602 // Windows 8
    #include <windows.h>
    #include <Processtopologyapi.h>
    #include <processthreadsapi.h>
#endif

#include "Math.tpp"
