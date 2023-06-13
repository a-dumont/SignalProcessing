#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdint>
#include <omp.h>
#include <future>

namespace py = pybind11;
using namespace pybind11::literals;

#include "Histograms_CUDA.h"

typedef py::array_t<float,py::array::c_style> np_float;
typedef py::array_t<double,py::array::c_style> np_double;
typedef py::array_t<uint8_t,py::array::c_style> np_uint8;
typedef py::array_t<uint16_t,py::array::c_style> np_uint16;
typedef py::array_t<uint32_t,py::array::c_style> np_uint32;
typedef py::array_t<uint64_t,py::array::c_style> np_uint64;
typedef py::array_t<int8_t,py::array::c_style> np_int8;
typedef py::array_t<int16_t,py::array::c_style> np_int16;
typedef py::array_t<int32_t,py::array::c_style> np_int32;
typedef py::array_t<int64_t,py::array::c_style> np_int64;

void dummyFree(void* ptr){}

#include "Histograms_CUDA_py.tpp"
