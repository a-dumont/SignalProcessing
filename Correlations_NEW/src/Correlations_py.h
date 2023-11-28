#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include<pybind11/numpy.h>
#include <stdexcept>

namespace py = pybind11;
using namespace pybind11::literals;

#include "Correlations.h"



 
// .tpp definitions
#include "Correlations_py.tpp"
