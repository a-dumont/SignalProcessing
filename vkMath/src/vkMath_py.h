#pragma once
#include "../../vkCompute/src/vulkanTools/vulkanTools_py.h"

namespace py = pybind11;
using namespace pybind11::literals;

// uint arrays
typedef py::array_t<uint64_t,py::array::c_style> uint64_a;
typedef py::array_t<uint32_t,py::array::c_style> uint32_a;
typedef py::array_t<uint16_t,py::array::c_style> uint16_a;
typedef py::array_t<uint8_t,py::array::c_style> uint8_a;

// Float arrays
typedef py::array_t<float,py::array::c_style> float32_a;
typedef py::array_t<double,py::array::c_style> float64_a;

// String array
typedef py::array_t<std::string,py::array::c_style> str_a;

// vkAdd
template<typename Datatype>
py::array_t<Datatype,py::array::c_style> vkAdd(vkComputer::Computer* computer, 
				ComputePipelinePy* pipeline,
				py::array_t<Datatype,py::array::c_style> in1, 
				py::array_t<Datatype,py::array::c_style> in2);

#include "vkMath_py.tpp"
