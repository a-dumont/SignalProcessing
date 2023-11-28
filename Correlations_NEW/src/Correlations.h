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


// Others
template<class DataTypeIn, class DataTypeOut>
void convertAVX(uint64_t N, DataTypeIn* in, DataTypeOut* out);

// .tpp template definitions
#include "Correlations.tpp"
