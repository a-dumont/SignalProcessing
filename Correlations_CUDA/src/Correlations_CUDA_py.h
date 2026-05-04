#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <cuda_runtime_api.h>
#include<cmath>
#include "../../FFT_CUDA/src/FFT_CUDA.h"

namespace py = pybind11;
using namespace pybind11::literals;

#include "Correlations_CUDA.h"

typedef py::array_t<double,py::array::c_style> np_double;
typedef py::array_t<std::complex<double>,py::array::c_style> np_complex;
typedef std::complex<double> dbl_complex;
typedef long long int llint_t;

void dummyFree(void* ptr){}

#include "Correlations_CUDA_py.tpp"
