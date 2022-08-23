#pragma once
#include <fftw3.h>
#include <string>
#include <stdio.h>
#include <cstdint>
#include <omp.h>

#ifdef _WIN64
	std::string wisdom_path = "FFTW_Wisdom";
	std::string wisdom_parallel_path = "FFTW_Parallel_Wisdom";
#else
	std::string wisdom_path = "/etc/FFTW/FFTW_Wisdom";
	std::string wisdom_parallel_path = "/etc/FFTW/FFTW_Parallel_Wisdom";
#endif

typedef std::complex<double> dbl_complex;
typedef std::complex<float> flt_complex;
	
#include "FFT.tpp"
