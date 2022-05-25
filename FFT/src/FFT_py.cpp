#include "FFT_py.h"
#include <complex>
#include <type_traits>

void init_fft(py::module &m)
{
	m.def("fft",&FFT_py<np_complex>, "in"_a);
	m.def("fft",&FFT_Parallel_py<np_complex>, "in"_a);
	m.def("ifft",&iFFT_py<np_complex>, "in"_a);
	m.def("fft",&FFT_py<np_complex,int>, "in"_a,"N"_a);
	m.def("rfft",&rFFT_py<np_double>, "in"_a.noconvert());
	m.def("rfft",&rFFT_py<np_double,int>, "in"_a.noconvert(),"N"_a);
	m.def("irfft",&irFFT_py<np_complex>, "in"_a.noconvert());
}

PYBIND11_MODULE(libfft, m)
{
	m.doc() = "Might work, might not.";
	init_fft(m);
}
