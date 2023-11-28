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

// Acorr
template<class DataType>
void aCorrFreqAVX(uint64_t N, DataType* in, DataType* out);

// Others
template<class DataTypeIn, class DataTypeOut>
void convertAVX(uint64_t N, DataTypeIn* in, DataTypeOut* out);

template<class DataTypeIn, class DataTypeOut>
void convertAVX_pad(uint64_t N, uint64_t Npad,
				DataTypeIn* in, DataTypeOut* out, DataTypeOut conv, DataTypeIn offset);

// .tpp template definitions 
#include "Correlations.tpp"
