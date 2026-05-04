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
void finite_difference_coefficients(int M, int N, DataType* coeff);

template<class DataType, class DataType2>
void nth_order_gradient(int n, DataType* x,
				DataType2 dt, DataType* out, int M, int N, DataType* coeff);

template<class DataType>
void continuous_max(long long int* out, DataType* in, int n);

template<class DataType>
void continuous_min(long long int* out, DataType* in, int n);

template<class DataType>
DataType sum_pairwise(DataType* in, long int n);

template<class DataType>
DataType variance_pairwise(DataType* in, long int n);

template<class DataType>
DataType skewness_pairwise(DataType* in, long int n);

template<class DataType, class DataType2>
void product(DataType* in1, DataType* in2, DataType2* out, int n);

template<class DataType, class DataType2>
void sum(DataType* in1, DataType* in2, DataType2* out, int n);

template<class DataType, class DataType2>
void difference(DataType* in1, DataType* in2, DataType2* out, int n);

template<class DataType, class DataType2>
void division(DataType* in1, DataType* in2, DataType2* out, int n);

template<class DataType>
DataType max(DataType* in, int n);

template<class DataType>
DataType min(DataType* in, int n);

template<class DataTypeIn, class DataTypeOut>
void block_max(int64_t N, int64_t block_size, DataTypeIn* in, DataTypeOut* out);

template<class DataTypeIn, class DataTypeOut>
void block_min(int64_t N, int64_t block_size, DataTypeIn* in, DataTypeOut* out);

template<class DataTypeIn, class DataTypeOut>
void block_min_max(int64_t N, int64_t block_size, DataTypeIn* in, DataTypeOut* out);

template<class DataTypeIn, class DataTypeOut>
void block_variance(int64_t N, int64_t block_size, DataTypeIn* in, DataTypeOut* out);




#include "Math.tpp"
