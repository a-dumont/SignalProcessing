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
aCorrCircFreqReduceAVX_py(py::array_t<DataType,py::array::c_style> py_in, uint64_t size);

template<class DataType>
py::array_t<DataType,py::array::c_style>
aCorrCircFreqReduceAVX_py(py::array_t<DataType,py::array::c_style> py_in, uint64_t size);

class ACorrCircularFreqAVX_py
{
	private:
		fftw_plan plan, plan2;
		fftwf_plan planf, plan2f;
		double *in, *out;
		float *inf, *outf;
		double **inThreads, **outThreads;
		float **inThreadsf, **outThreadsf;
		uint64_t N, size, cSize, howmany, Npad, threads, howmanyPerThread;
		int length[1];
		uint64_t transferSize;

	public:
		ACorrCircularFreqAVX_py(uint64_t N_in, uint64_t size_in);
		~ACorrCircularFreqAVX_py();
	
		void train();
		std::tuple<double,double> benchmark(uint64_t n);
		py::array_t<double,1> aCorrCircularFreqAVX(py::array_t<double,1> py_in);
		py::array_t<float,1> aCorrCircularFreqAVXf(py::array_t<float,1> py_in);

		uint64_t getSize();
		uint64_t getN();
		uint64_t getHowmany();
};

class DigitizerACorrCircularFreqAVX_py
{
	private:
		fftw_plan plan, plan2;
		fftwf_plan planf, plan2f;
		double *in, *out;
		float *inf, *outf;
		double **inThreads, **outThreads;
		float **inThreadsf, **outThreadsf;
		uint64_t N, size, cSize, howmany, Npad, threads, howmanyPerThread;
		int length[1];
		uint64_t transferSize;
	
	public:
		DigitizerACorrCircularFreqAVX_py(uint64_t N_in, uint64_t size_in);
		~DigitizerACorrCircularFreqAVX_py();
	
		void train();
		std::tuple<double,double> benchmark(uint64_t n);

		template<class DataType>
		py::array_t<double,1> 
		aCorrCircularFreqAVX(py::array_t<DataType,1> py_in, double conv, DataType offset);

		template<class DataType>
		py::array_t<float,1> 
		aCorrCircularFreqAVXf(py::array_t<DataType,1> py_in, float conv, DataType offset);

		uint64_t getSize();
		uint64_t getN();
		uint64_t getHowmany();
};

// Xcorr
template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style>
xCorrCircularFreq(py::array_t<DataType,py::array::c_style> py_in1, 
				py::array_t<DataType,py::array::c_style> py_in2, uint64_t size);

class XCorrCircularFreqAVX_py
{
	private:
		fftw_plan plan, plan2;
		fftwf_plan planf, plan2f;
		double *in, *out1, *out2;
		float *inf, *out1f, *out2f;
		double **inThreads, **outThreads1, **outThreads2;
		float **inThreadsf, **outThreads1f, **outThreads2f;
		uint64_t N, size, cSize, howmany, Npad, threads, howmanyPerThread;
		int length[1];
		uint64_t transferSize;

	public:
		XCorrCircularFreqAVX_py(uint64_t N_in, uint64_t size_in);
		~XCorrCircularFreqAVX_py();

		void train();
		std::tuple<double,double> benchmark(uint64_t n);

		py::array_t<std::complex<double>,1> 
		xCorrCircularFreqAVX(py::array_t<double,1> py_in1, py::array_t<double,1> py_in2);
		
		py::array_t<std::complex<float>,1> 
		xCorrCircularFreqAVXf(py::array_t<float,1> py_in1, py::array_t<float,1> py_in2);

		uint64_t getSize();
		uint64_t getN();
		uint64_t getHowmany();
};

class DigitizerXCorrCircularFreqAVX_py
{
	private:
		fftw_plan plan, plan2;
		fftwf_plan planf, plan2f;
		double *in, *out1, *out2;
		float *inf, *out1f, *out2f;
		double **inThreads, **outThreads1, **outThreads2;
		float **inThreadsf, **outThreads1f, **outThreads2f;
		uint64_t N, size, cSize, howmany, Npad, threads, howmanyPerThread;
		int length[1];
		uint64_t transferSize;

	public:
		DigitizerXCorrCircularFreqAVX_py(uint64_t N_in, uint64_t size_in);
		~DigitizerXCorrCircularFreqAVX_py();

		void train();
		std::tuple<double,double> benchmark(uint64_t n);

		template<class DataType>
		py::array_t<std::complex<double>,1> 
		xCorrCircularFreqAVX
		(py::array_t<DataType,1>py_in1,py::array_t<DataType,1>py_in2,double conv,DataType offset);

		template<class DataType>
		py::array_t<std::complex<float>,1> 
		xCorrCircularFreqAVXf
		(py::array_t<DataType,1>py_in1,py::array_t<DataType,1>py_in2,float conv,DataType offset);

		uint64_t getSize();
		uint64_t getN();
		uint64_t getHowmany();
};

// Combined Acorr and Xcorr
template<class DataType>
std::tuple<py::array_t<DataType,py::array::c_style>,
py::array_t<DataType,py::array::c_style>,
py::array_t<std::complex<DataType>,py::array::c_style>>
fCorrCircFreqReduceAVX_py(py::array_t<DataType,py::array::c_style> py_in1, 
				py::array_t<DataType,py::array::c_style> py_in2, uint64_t size);

class FCorrCircularFreqAVX_py
{
	private:
		fftw_plan plan, plan2;
		fftwf_plan planf, plan2f;
		double *in, *out1, *out2;
		float *inf, *out1f, *out2f;
		double **inThreads, **outThreads1, **outThreads2;
		float **inThreadsf, **outThreads1f, **outThreads2f;
		uint64_t N, size, cSize, howmany, Npad, threads, howmanyPerThread;
		int length[1];
		uint64_t transferSize;

	public:
		FCorrCircularFreqAVX_py(uint64_t N_in, uint64_t size_in);
		~FCorrCircularFreqAVX_py();

		void train();
		std::tuple<double,double> benchmark(uint64_t n);

		std::tuple<py::array_t<double,1>,py::array_t<double,1>,py::array_t<std::complex<double>,1>>
		fCorrCircularFreqAVX(py::array_t<double,1> py_in1, py::array_t<double,1> py_in2);
		
		std::tuple<py::array_t<float,1>,py::array_t<float,1>,py::array_t<std::complex<float>,1>>
		fCorrCircularFreqAVXf(py::array_t<float,1> py_in1, py::array_t<float,1> py_in2);

		uint64_t getSize();
		uint64_t getN();
		uint64_t getHowmany();
};


class DigitizerFCorrCircularFreqAVX_py
{
	private:
		fftw_plan plan, plan2;
		fftwf_plan planf, plan2f;
		double *in, *out1, *out2, *out3;
		float *inf, *out1f, *out2f, *out3f;
		double **inThreads, **outThreads1, **outThreads2, **outThreads3;
		float **inThreadsf, **outThreads1f, **outThreads2f, **outThreads3f;
		uint64_t N, size, cSize, howmany, Npad, threads, howmanyPerThread;
		int length[1];
		uint64_t transferSize;

	public:
		DigitizerFCorrCircularFreqAVX_py(uint64_t N_in, uint64_t size_in);
		~DigitizerFCorrCircularFreqAVX_py();

		void train();
		std::tuple<double,double> benchmark(uint64_t n);

		template<class DataType>
		std::tuple<py::array_t<double,1>,py::array_t<double,1>,py::array_t<std::complex<double>,1>>
		fCorrCircularFreqAVX
		(py::array_t<DataType,1>py_in1,py::array_t<DataType,1>py_in2,double conv,DataType offset);

		template<class DataType>
		std::tuple<py::array_t<float,1>,py::array_t<float,1>,py::array_t<std::complex<float>,1>>
		fCorrCircularFreqAVXf
		(py::array_t<DataType,1>py_in1,py::array_t<DataType,1>py_in2,float conv,DataType offset);

		uint64_t getSize();
		uint64_t getN();
		uint64_t getHowmany();
};
 
// .tpp definitions
#include "Correlations_py.tpp"
