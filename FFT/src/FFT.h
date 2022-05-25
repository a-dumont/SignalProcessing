#pragma once
#include <fftw3.h>
#include <string>
#include <stdio.h>

#ifdef _WIN64
	std::string wisdom_path = "FFTW_Wisdom";
	std::string wisdom_parallel_path = "FFTW_Parallel_Wisdom";
#else
	std::string wisdom_path = "/etc/FFTW/FFTW_Wisdom";
	std::string wisdom_parallel_path = "/etc/FFTW/FFTW_Parallel_Wisdom";
#endif

#include "FFT.tpp"
