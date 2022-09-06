#pragma once
#include <cuda.h>
#include <cuComplex.h>
#include <complex>
#include <cmath>

template<class DataType>
void autocorrelation_cuda(long long int N, std::complex<DataType>* in, DataType* out);

template<class DataType>
void cross_correlation_cuda(long long int N, std::complex<DataType>* in1, 
				std::complex<DataType>*in2, std::complex<DataType>* out);

template<class DataType>
void complete_correlation_cuda(long long int N, std::complex<DataType>* in1, 
				std::complex<DataType>*in2, 
				DataType* out1, 
				DataType* out2,
				std::complex<DataType>* out3);

template<class DataType> 
void reduction(long long int N, DataType* in, long long int size);

template<class DataType> 
void reduction_general(long long int N, DataType* in, long long int size);

template<class DataType, class DataType2>
void convertComplex(long long int N, DataType* in, std::complex<DataType2>* out, 
				DataType2 conv, DataType offset, cudaStream_t stream);

template<class DataType, class DataType2>
void convert(long long int N, DataType* in, DataType2* out, 
				DataType2 conv, DataType offset, cudaStream_t stream);
