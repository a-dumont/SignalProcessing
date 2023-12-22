#pragma once
#include <fftw3.h>
#include <string>
#include <stdio.h>
#include <cstdint>
#include <complex>
#include <omp.h>
#include <type_traits>
#include <chrono>
#include <immintrin.h>
#include <cstring>
#include <numeric>

// Acorr
template<class DataType>
void aCorrCircularFreqAVX(uint64_t N, DataType* in, DataType* out);

// Xcorr
template<class DataType>
void xCorrCircularFreqAVX(uint64_t N, DataType* in1, DataType* in2, DataType* out);

// Combined Acorr and Xcorr
template<class DataType>
void axCorrCircularFreqAVX(uint64_t N, DataType* in1, DataType* in2, 
				DataType* out1, DataType* out2, DataType* out3);

// rFFT
template<class DataType>
void rfftBlock(int N, int size, DataType* in, std::complex<DataType>* out);

// Type conversion
template<class DataTypeIn, class DataTypeOut>
void convertAVX(uint64_t N, DataTypeIn* in, DataTypeOut* out);

template<class DataTypeIn, class DataTypeOut>
void convertAVX_pad(uint64_t N, uint64_t Npad,
				DataTypeIn* in, DataTypeOut* out, DataTypeOut conv, DataTypeIn offset);
				
// Reduction
template<class DataType>
void reduceAVX(uint64_t N, DataType* in, DataType* out);

template<class DataType>
void reduceBlockAVX(uint64_t N, uint64_t size, DataType* in, DataType* out);

// .tpp template definitions 
#include "Correlations.tpp"