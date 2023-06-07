#pragma once
#include <pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include <stdexcept>
#include <cstdint>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

#include "Histograms.h"

typedef py::array_t<uint64_t,py::array::c_style> np_uint64;
typedef py::array_t<int64_t,py::array::c_style> np_int64;
typedef py::array_t<uint32_t,py::array::c_style> np_uint32;
typedef py::array_t<uint16_t,py::array::c_style> np_uint16;

typedef py::array_t<float,py::array::c_style> np_float;
typedef py::array_t<double,py::array::c_style> np_double;

#include "Histograms_py.tpp"
