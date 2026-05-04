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

#include "Math_CUDA.h"

#include "Math_CUDA_py.tpp"
