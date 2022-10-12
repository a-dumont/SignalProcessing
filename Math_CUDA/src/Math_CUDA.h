#pragma once
#include <cuda.h>
#include <cuComplex.h>
#include <complex>
#include <cmath>
#include <cstdint>

typedef std::complex<double> dbl_complex;
typedef long long int llint_t;

template <class DataType>
void vector_sum(llint_t N, DataType in1, DataType in2);

template <class DataType>
void vector_product(llint_t N, DataType in1, DataType in2);

template <class DataType>
void vector_diff(llint_t N, DataType in1, DataType in2);

template <class DataType>
void vector_div(llint_t N, DataType in1, DataType in2);

template <class DataType>
void martrix_sum(llint_t Nr, llint_t Nc, DataType in1, DataType in2);

template <class DataType>
void martrix_prod(llint_t Nr, llint_t Nc, DataType in1, DataType in2);

template <class DataType>
void martrix_diff(llint_t Nr, llint_t Nc, DataType in1, DataType in2);

template <class DataType>
void martrix_div(llint_t Nr, llint_t Nc, DataType in1, DataType in2);

template <class DataType>
void gradient(llint_t N, DataType* in, double* out, double h);