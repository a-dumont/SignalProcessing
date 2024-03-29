#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include<pybind11/numpy.h>
#include <stdexcept>

namespace py = pybind11;
using namespace pybind11::literals;

#include "Filters.h"

typedef py::array_t<long long int,py::array::c_style> np_int;
typedef py::array_t<uint64_t,py::array::c_style> np_uint;
typedef py::array_t<double,py::array::c_style> np_double;
typedef py::array_t<std::complex<double>,py::array::c_style> np_complex;
typedef std::complex<double> dbl_complex; 

#include "Filters_py.tpp"
