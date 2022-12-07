#pragma once
#include <cmath>
#include <cstdint>
#include <cuda.h>

template<class DataType>
void filter_CUDA(long long int N, DataType* signal, DataType* filter, cudaStream_t stream);

template<class DataType, class DataType2>
void convert(long long int N, DataType* in, DataType2* out,
				DataType2 conv, DataType offset, cudaStream_t stream);

template<class DataType, class DataType2>
void rconvert(long long int N, DataType2* in, DataType* out,
								DataType2 conv, DataType offset, cudaStream_t stream);
