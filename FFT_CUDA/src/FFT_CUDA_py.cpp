#include "FFT_CUDA_py.h"
#include <complex>
#include <type_traits>

void init_fft(py::module &m)
{
	// Double
	m.def("fft_cuda",&FFT_CUDA_py<np_complex>, "in"_a);
	m.def("fft_cuda",&FFT_Block_CUDA_py<np_complex,long long int>, "in"_a,"N"_a);
	m.def("ifft_cuda",&iFFT_CUDA_py<np_complex>, "in"_a);
	m.def("rfft_cuda",&rFFT_CUDA_py<np_double>, "in"_a.noconvert());
	m.def("rfft_cuda",&rFFT_Block_CUDA_py<np_double,int>, "in"_a.noconvert(),"N"_a);
	m.def("irfft_cuda",&irFFT_CUDA_py<np_complex>, "in"_a.noconvert());
	// Float
	m.def("ffft_cuda",&fFFT_CUDA_py<np_fcomplex>, "in"_a);
	m.def("ffft_cuda",&fFFT_Block_CUDA_py<np_fcomplex,long long int>, "in"_a,"N"_a);
	m.def("iffft_cuda",&ifFFT_CUDA_py<np_fcomplex>, "in"_a);
	m.def("rffft_cuda",&rfFFT_CUDA_py<np_float>, "in"_a.noconvert());
	m.def("rffft_cuda",&rfFFT_Block_CUDA_py<np_float,int>, "in"_a.noconvert(),"N"_a);
	m.def("irffft_cuda",&irfFFT_CUDA_py<np_fcomplex>, "in"_a.noconvert());
}

PYBIND11_MODULE(libfftcuda, m)
{
	m.doc() = "Might work, might not.";
	init_fft(m);
}
