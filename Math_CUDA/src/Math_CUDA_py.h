#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdint>

namespace py = pybind11;
using namespace pybind11::literals;

typedef std::complex<double> dbl_complex;
typedef long long int llint_t;

#include "Math_CUDA_py.tpp"