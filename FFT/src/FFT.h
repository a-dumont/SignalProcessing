#pragma once
#include <fftw3.h>
#include <string>
#include <stdio.h>

#ifdef _WIN64
	std::string wisdom_path = "FFTW_Wisdom";
#else
	std::string wisdom_path = "/etc/FFTW/FFTW_Wisdom";
#endif

#include "FFT.tpp"
