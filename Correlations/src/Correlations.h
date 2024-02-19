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

#if defined(__CYGWIN__) || defined(__MINGW64__)
    #define _WIN32_WINNT 0x0602 // Windows 8
    #include <windows.h>
    #include <Processtopologyapi.h>
    #include <processthreadsapi.h>
#endif

#ifdef _WIN64
	std::string wisdom_path = "FFTW_Wisdom";
	std::string wisdom_pathf = "FFTW_Wisdomf";
#else
	std::string wisdom_path = "/etc/FFTW/FFTW_Wisdom";
	std::string wisdom_pathf = "/etc/FFTW/FFTW_Wisdomf";;
#endif

typedef std::chrono::steady_clock Clock;

void manage_thread_affinity();

// Acorr
template<class DataType>
void aCorrCircFreqReduceAVX(uint64_t N, uint64_t size, DataType* data);

// Xcorr
template<class DataType>
void xCorrCircFreqReduceAVX(uint64_t N, uint64_t size, DataType* data1, DataType* data2);

// Combined Acorr and Xcorr
template<class DataType>
void fCorrCircFreqReduceAVX(uint64_t N, uint64_t size, DataType* data1, DataType* data2);

// rFFT
template<class DataType>
void rfftBlock(int N, int size, DataType* in, std::complex<DataType>* out);

// Type conversion
template<class DataTypeIn, class DataTypeOut>
void convertAVX(uint64_t N, DataTypeIn* in, DataTypeOut* out, DataTypeOut conv, DataTypeIn offset);

template<class DataTypeIn, class DataTypeOut>
void convertAVX_pad(uint64_t N, uint64_t Npad,
				DataTypeIn* in, DataTypeOut* out, DataTypeOut conv, DataTypeIn offset);
				
// Reduction
template<class DataType>
void reduceAVX(uint64_t N, DataType* in, DataType* out);

template<class DataType>
void reduceInPlaceBlockAVX(uint64_t N, uint64_t size, DataType* in);

// .tpp template definitions 
#include "Correlations.tpp"
