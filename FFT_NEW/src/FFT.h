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

#ifdef _WIN64
	std::string wisdom_path = "FFTW_Wisdom";
	std::string wisdom_pathf = "FFTW_Wisdomf";
#else
	std::string wisdom_path = "/etc/FFTW/FFTW_Wisdom";
	std::string wisdom_pathf = "/etc/FFTW/FFTW_Wisdomf";
#endif

typedef std::chrono::steady_clock Clock;

void manage_thread_affinity();

// FFT
template<class DataType>
void fft(uint64_t N, std::complex<DataType>* in, std::complex<DataType>* out);

template<class DataType>
void fft_training(uint64_t N, std::complex<DataType>* in, std::complex<DataType>* out);

template<class DataType>
void fftBlock(int N, int size, std::complex<DataType>* in, std::complex<DataType>* out);

template<class DataType>
void fftBlock_training(int N, int size, std::complex<DataType>* in, std::complex<DataType>* out);

// rFFT
template<class DataType>
void rfft(uint64_t N, DataType* in, std::complex<DataType>* out);

template<class DataType>
void rfft_training(uint64_t N, DataType* in, std::complex<DataType>* out);

template<class DataType>
void rfftBlock(int N, int size, DataType* in, std::complex<DataType>* out);

template<class DataType>
void rfftBlock_training(int N, int size, DataType* in, std::complex<DataType>* out);

template<class DataType>
void rfftBlock_inplace(int N, int size, 
	DataType* in, std::complex<DataType>* out);

template<class DataType>
void rfftBlock_inplace_training(int N, int size, 
	DataType* in, std::complex<DataType>* out);

// iFFT
template<class DataType>
void ifft(uint64_t N, std::complex<DataType>* in, std::complex<DataType>* out);

template<class DataType>
void ifft_training(uint64_t N, std::complex<DataType>* in, std::complex<DataType>* out);

template<class DataType>
void ifftBlock(int N, int size, std::complex<DataType>* in, std::complex<DataType>* out);

template<class DataType>
void ifftBlock_training(int N, int size, std::complex<DataType>* in, std::complex<DataType>* out);

// irFFT
template<class DataType>
void irfft(uint64_t N, std::complex<DataType>* in, DataType* out);

template<class DataType>
void irfft_training(uint64_t N, std::complex<DataType>* in, DataType* out);

template<class DataType>
void irfftBlock(int N, int size, std::complex<DataType>* in, DataType* out);

template<class DataType>
void irfftBlock_training(int N, int size, std::complex<DataType>* in, DataType* out);

// Others
template<class DataTypeIn, class DataTypeOut>
void convertAVX(uint64_t N, DataTypeIn* in, DataTypeOut* out);

template<class DataTypeIn, class DataTypeOut>
void convertAVX_pad(uint64_t N, uint64_t Npad, 
				DataTypeIn* in, DataTypeOut* out, DataTypeOut conv, DataTypeIn offset);

// .tpp definitions
#include "FFT.tpp"
