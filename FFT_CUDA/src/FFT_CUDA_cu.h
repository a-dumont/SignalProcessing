#pragma once
#include <cuda.h>
#include<cuda_runtime_api.h>
#include <cuComplex.h>
#include <complex>

template<class DataType, class DataType2>
void convertComplex(long long int N, DataType* in, std::complex<DataType2>* out, 
				DataType2 conv, DataType offset, cudaStream_t stream);

template<class DataType, class DataType2>
void convert(long long int N, DataType* in, DataType2* out, 
				DataType2 conv, DataType offset, cudaStream_t stream);
