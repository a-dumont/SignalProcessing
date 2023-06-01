#pragma once
#include <cuda.h>
#include <cmath>
#include <cstdint>

template<class DataType> 
void reduction(uint64_t N, DataType* in, uint64_t size);

template<class DataType> 
void reduction_general(uint64_t N, DataType* in, uint64_t size);

template<class DataType>
void digitizer_histogram_1d(uint64_t N, DataType* in, uint32_t* hist, cudaStream_t stream);

template<class DataType>
void digitizer_histogram_subbyte_1d(uint64_t N, DataType* in, uint32_t* hist, 
				uint8_t nbits, cudaStream_t stream);

template<class DataType>
void digitizer_histogram_2d(uint64_t N, DataType* in_x, DataType* in_y, uint32_t* hist, 
				cudaStream_t stream);

template<class DataType>
void digitizer_histogram_subbyte_2d(uint64_t N, DataType* in_x, DataType* in_y, uint32_t* hist, 
				uint8_t nbits, cudaStream_t stream);
