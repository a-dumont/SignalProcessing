#include "FFT_CUDA_py.h"

void init_fft(py::module &m)
{
	m.def("fft_cuda",&FFT_CUDA_py<float>,"in"_a);
	m.def("fft_cuda",&FFT_CUDA_py<double>,"in"_a);	
	m.def("fft_cuda",&FFT_Block_Async_CUDA_py<float,long long int>, "in"_a,"N"_a);
	m.def("fft_cuda",&FFT_Block_Async_CUDA_py<double,long long int>, "in"_a,"N"_a);

	m.def("ifft_cuda",&iFFT_CUDA_py<float>,"in"_a);	
	m.def("ifft_cuda",&iFFT_CUDA_py<double>,"in"_a);	

	m.def("rfft_cuda",&rFFT_CUDA_py<float>, "in"_a.noconvert());
	m.def("rfft_cuda",&rFFT_CUDA_py<double>, "in"_a.noconvert());
	m.def("rfft_cuda",&rFFT_Block_CUDA2_py<float,long long int>, "in"_a.noconvert(),"N"_a);
	m.def("rfft_cuda",&rFFT_Block_CUDA2_py<double,long long int>, "in"_a.noconvert(),"N"_a);

	m.def("irfft_cuda",&irFFT_CUDA_py<float>, "in"_a.noconvert());
	m.def("irfft_cuda",&irFFT_CUDA_py<double>, "in"_a.noconvert());

	m.def("digitizer_FFT_cuda",&digitizer_FFT_Block_Async_CUDA_py<uint8_t,double>,
					"in"_a.noconvert(),"size"_a,"conv"_a.noconvert(),"offset"_a);
	m.def("digitizer_FFT_cudaf",&digitizer_FFT_Block_Async_CUDA_py<uint8_t,float>,
					"in"_a.noconvert(),"size"_a,"conv"_a.noconvert(),"offset"_a);

	m.def("digitizer_rFFT_cuda",&digitizer_rFFT_Block_Async_CUDA_py<uint8_t,double>,
					"in"_a.noconvert(),"size"_a,"conv"_a.noconvert(),"offset"_a);
	m.def("digitizer_rFFT_cudaf",&digitizer_rFFT_Block_Async_CUDA_py<uint8_t,float>,
					"in"_a.noconvert(),"size"_a,"conv"_a.noconvert(),"offset"_a);
}

PYBIND11_MODULE(libfftcuda, m)
{
	m.doc() = "Might work, might not.";
	init_fft(m);
}
