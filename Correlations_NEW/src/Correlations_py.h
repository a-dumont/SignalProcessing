#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include<pybind11/numpy.h>
#include <stdexcept>

namespace py = pybind11;
using namespace pybind11::literals;

#include "Correlations.h"

// Acorr
template<class DataType>
py::array_t<DataType,py::array::c_style>
aCorrCircularFreqAVX_py(py::array_t<DataType,py::array::c_style> py_in, uint64_t size);

// Xcorr
template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style>
xCorrCircularFreqAVX_py(py::array_t<DataType,py::array::c_style> py_in1, 
				py::array_t<DataType,py::array::c_style> py_in2, uint64_t size);

// Combined Acorr and Xcorr
template<class DataType>
std::tuple<py::array_t<DataType,py::array::c_style>,
py::array_t<DataType,py::array::c_style>,
py::array_t<std::complex<DataType>,py::array::c_style>>
axCorrCircularFreqAVX_py(py::array_t<DataType,py::array::c_style> py_in1, 
				py::array_t<DataType,py::array::c_style> py_in2, uint64_t size);

// Reduction
template<class DataType>
DataType reduceAVX_py(py::array_t<DataType,py::array::c_style> py_in);

template<class DataType>
py::array_t<DataType,py::array::c_style>
reduceBlockAVX_py(py::array_t<DataType,py::array::c_style> py_in, uint64_t size);
 
// .tpp definitions
#include "Correlations_py.tpp"