#pragma once
#include <cufft.h>
#include <complex>
#include <string>
#include <stdio.h>

typedef std::complex<double> dbl_complex; 
typedef std::complex<float> flt_complex; 

template<class DataType, class DataType2>
void convertComplex(long long int N, DataType* in, std::complex<DataType2>* out, 
				DataType2 conv, DataType offset, cudaStream_t stream);

template<class DataType, class DataType2>
void convert(long long int N, DataType* in, DataType2* out, 
				DataType2 conv, DataType offset, cudaStream_t stream);

#include "FFT_CUDA.tpp"
