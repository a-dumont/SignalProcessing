#pragma once
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <omp.h>
#include <stdexcept>
#include <cufft.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#if defined(__CYGWIN__) || defined(__MINGW64__)
    #define _WIN32_WINNT 0x0602 // Windows 8
    #include <windows.h>
    #include <Processtopologyapi.h>
    #include <processthreadsapi.h>
#endif

template<class DataType>
void filter_CUDA(long long int N, DataType* signal, DataType* filter, cudaStream_t stream);

template<class DataType, class DataType2>
void convert(long long int N, DataType* in, DataType2* out,
				DataType2 conv, DataType offset, cudaStream_t stream);

template<class DataType, class DataType2>
void rconvert(long long int N, DataType2* in, DataType* out,
								DataType2 conv, DataType offset, cudaStream_t stream);

#include "Histograms_CUDA.tpp"
