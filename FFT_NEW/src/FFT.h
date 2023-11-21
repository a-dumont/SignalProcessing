#pragma once
#include <fftw3.h>
#include <string>
#include <stdio.h>
#include <cstdint>
#include <complex>
#include <omp.h>
#include <type_traits>
#include <chrono>

#ifdef _WIN64
	std::string wisdom_path = "FFTW_Wisdom";
#else
	std::string wisdom_path = "/etc/FFTW/FFTW_Wisdom";
#endif

typedef std::chrono::steady_clock Clock;
	
template<class DataType>
void fft(uint64_t N, std::complex<DataType>* in, std::complex<DataType>* out);

template<class DataType>
void fft_training(uint64_t N, std::complex<DataType>* in, std::complex<DataType>* out);

template<class DataType>
void rfft(uint64_t N, DataType* in, std::complex<DataType>* out);

template<class DataType>
void rfft_training(uint64_t N, DataType* in, std::complex<DataType>* out);

template<class DataType>
void ifft(uint64_t N, std::complex<DataType>* in, std::complex<DataType>* out);

template<class DataType>
void ifft_training(uint64_t N, std::complex<DataType>* in, std::complex<DataType>* out);

template<class DataType>
void irfft(uint64_t N, std::complex<DataType>* in, DataType* out);

template<class DataType>
void irfft_training(uint64_t N, std::complex<DataType>* in, DataType* out);

#include "FFT.tpp"