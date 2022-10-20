#pragma once
#include <tuple>
#include <cmath>
#include <stdio.h>
#include <algorithm>
#include <cstdint>
#include <omp.h>

#if defined(__CYGWIN__) || defined(__MINGW64__)
    #define _WIN32_WINNT 0x0602 // Windows 8
    #include <windows.h>
    #include <Processtopologyapi.h>
    #include <processthreadsapi.h>
#endif

#include "../../Math/src/Math.h"
#include "Histograms.tpp"
