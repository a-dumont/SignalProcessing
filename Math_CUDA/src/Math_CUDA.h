#pragma once
#include <cuda.h>
#include <cuComplex.h>
#include <complex>
#include <cmath>
#include <cstdint>

typedef std::complex<double> dbl_complex;

template <class DataType>
void vector_sum(long long int N, DataType* in1, DataType* in2);

template <class DataType>
void vector_product(long long int N, DataType* in1, DataType* in2);

template <class DataType>
void vector_diff(long long int N, DataType* in1, DataType* in2);

template <class DataType>
void vector_div(long long int N, DataType* in1, DataType* in2);

template <class DataType>
void matrix_sum(long long int Nr, long long int Nc, DataType* in1, DataType* in2);

template <class DataType>
void matrix_prod(long long int Nr, long long int Nc, DataType* in1, DataType* in2);

template <class DataType>
void matrix_diff(long long int Nr, long long int Nc, DataType* in1, DataType* in2);

template <class DataType>
void matrix_div(long long int Nr, long long int Nc, DataType* in1, DataType* in2);

template <class DataType>
void gradient(long long int N, DataType* in, double* out, double h);
