#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <complex>

#include "FFT.h"

namespace py = pybind11;
using namespace pybind11::literals;

// FFT
template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
fft_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in);

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
fft_training_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in);

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
fft_pad_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in, uint64_t size);

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style>
fftBlock_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in, uint64_t size);

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style>
fftBlock_training_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in, uint64_t size);

class FFT_py;

class FFT_Block_py;

// rFFT
template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
rfft_py(py::array_t<DataType,py::array::c_style> py_in);

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
rfft_training_py(py::array_t<DataType,py::array::c_style> py_in);

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
rfft_pad_py(py::array_t<DataType,py::array::c_style> py_in, uint64_t size);

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style>
rfftBlock_py(py::array_t<DataType,py::array::c_style> py_in, uint64_t size);

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style>
rfftBlock_training_py(py::array_t<DataType,py::array::c_style> py_in, uint64_t size);

template<class DataTypeIn, class DataTypeOut>
py::array_t<std::complex<DataTypeOut>,py::array::c_style> 
digitizer_rfft_py(py::array_t<DataTypeIn,py::array::c_style> py_in, 
				DataTypeOut conv, DataTypeIn offset);

template<class DataTypeIn, class DataTypeOut>
py::array_t<std::complex<DataTypeOut>,py::array::c_style>
digitizer_rfftBlock_py(py::array_t<DataTypeIn,py::array::c_style> py_in, 
				uint64_t size, DataTypeOut conv, DataTypeIn offset);

class RFFT_py;

class RFFT_Block_py;

// iFFT
template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
ifft_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in);

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
ifft_training_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in);

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
ifft_pad_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in, uint64_t size);

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style>
ifftBlock_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in, uint64_t size);

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style>
ifftBlock_training_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in, uint64_t size);

class IFFT_py;

class IFFT_Block_py;

// irFFT
template<class DataType>
py::array_t<DataType,py::array::c_style> 
irfft_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in);

template<class DataType>
py::array_t<DataType,py::array::c_style> 
irfft_training_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in);

template<class DataType>
py::array_t<DataType,py::array::c_style> 
irfft_pad_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in, uint64_t size);

template<class DataType>
py::array_t<DataType,py::array::c_style>
irfftBlock_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in, uint64_t size);

template<class DataType>
py::array_t<DataType,py::array::c_style>
irfftBlock_training_py(py::array_t<std::complex<DataType>,1> py_in, uint64_t size);

class IRFFT_py;

// Others
template<class DataTypeIn, class DataTypeOut>
py::array_t<DataTypeOut,py::array::c_style>
convertAVX_py(py::array_t<DataTypeIn,py::array::c_style>, DataTypeOut conv, DataTypeIn offset);

// .tpp definitions
#include "FFT_py.tpp"
