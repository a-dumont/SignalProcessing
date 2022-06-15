#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include<pybind11/numpy.h>
#include <stdexcept>
#include <cuda_runtime_api.h>

namespace py = pybind11;
using namespace pybind11::literals;

#include "FFT_CUDA.h"

typedef py::array_t<double,py::array::c_style> np_double;
typedef py::array_t<std::complex<double>,py::array::c_style> np_complex;

typedef py::array_t<float,py::array::c_style> np_float;
typedef py::array_t<std::complex<float>,py::array::c_style> np_fcomplex;

void cuFree(void* ptr)
{
	cudaFree(ptr);
}
void cuFreeHost(void* ptr)
{
	cudaFreeHost(ptr);
}

#include "FFT_CUDA_py.tpp"
