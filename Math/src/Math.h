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


void manage_thread_affinity();

template<class DataType, class DataType2>
void gradient(int n, DataType* x, DataType2* t, DataType* out);

template<class DataType, class DataType2>
void gradient2(int n, DataType* x, DataType2 dt, DataType* out);

template<class DataType>
void finite_difference_coefficients(uint64_t M, uint64_t N, DataType* coeff);

template<class DataType, class DataType2>
void nth_order_gradient(int n, DataType* x,
				DataType2 dt, DataType* out, uint64_t M, uint64_t N, DataType* coeff);

template<class DataType>
void continuous_max(uint64_t n, DataType* in, uint64_t* out);

template<class DataType>
void continuous_min(uint64_t n, DataType* in, uint64_t* out);

template<class DataType>
DataType sum_pairwise(uint64_t n, DataType* in);

template<class DataType>
DataType variance_pairwise(uint64_t n, DataType* in);

template<class DataType>
DataType skewness_pairwise(uint64_t n, DataType* in);

template<class DataType, class DataType2>
void product(uint64_t n, DataType* in1, DataType* in2, DataType2* out);

template<class DataType, class DataType2>
void sum(uint64_t n, DataType* in1, DataType* in2, DataType2* out);

template<class DataType, class DataType2>
void difference(uint64_t n, DataType* in1, DataType* in2, DataType2* out);

template<class DataType, class DataType2>
void division(uint64_t n, DataType* in1, DataType* in2, DataType2* out);

template<class DataType>
DataType max(uint64_t n, DataType* in);

template<class DataType>
DataType min(uint64_t n, DataType* in);

template<class DataTypeIn, class DataTypeOut>
void block_max(uint64_t N, uint64_t block_size, DataTypeIn* in, DataTypeOut* out);

template<class DataTypeIn, class DataTypeOut>
void block_min(uint64_t N, uint64_t block_size, DataTypeIn* in, DataTypeOut* out);

template<class DataTypeIn, class DataTypeOut>
void block_min_max(uint64_t N, uint64_t block_size, DataTypeIn* in, DataTypeOut* out);

template<class DataTypeIn, class DataTypeOut>
void block_variance(uint64_t N, uint64_t block_size, DataTypeIn* in, DataTypeOut* out);




#include "Math.tpp"
