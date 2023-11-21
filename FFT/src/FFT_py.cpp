#include "FFT_py.h"
#include <complex>
#include <type_traits>

void init_fft(py::module &m)
{
	m.def("fft",&FFT_py<double>, "in"_a.noconvert());
	m.def("fft",&FFT_py<float>, "in"_a.noconvert());
	m.def("fft_train",&FFT_train_py<double>, "in"_a.noconvert());
	m.def("fft_train",&FFT_train_py<float>, "in"_a.noconvert());
	
	m.def("ifft",&iFFT_py<double>, "in"_a.noconvert());
	m.def("ifft",&iFFT_py<float>, "in"_a.noconvert());
	m.def("ifft_train",&iFFT_train_py<double>, "in"_a.noconvert());
	m.def("ifft_train",&iFFT_train_py<float>, "in"_a.noconvert());
	
	m.def("fft",&FFT_py<double,int>, "in"_a.noconvert(),"N"_a);
	m.def("fft",&FFT_py<float,int>, "in"_a.noconvert(),"N"_a);
	m.def("fft_train",&FFT_train_py<double,int>, "in"_a.noconvert(),"N"_a);
	m.def("fft_train",&FFT_train_py<float,int>, "in"_a.noconvert(),"N"_a);
	
	//m.def("fft_parallel",&FFT_Block_Parallel_py<double,int>, "in"_a.noconvert(),"N"_a,"nthreads"_a);
	//m.def("fft_parallel",&FFT_Block_Parallel_py<float,int>, "in"_a.noconvert(),"N"_a,"nthreads"_a);
	
	m.def("rfft",&rFFT_py<double>, "in"_a.noconvert());
	m.def("rfft",&rFFT_py<float>, "in"_a.noconvert());
	m.def("rfft_train",&rFFT_train_py<double>, "in"_a.noconvert());
	m.def("rfft_train",&rFFT_train_py<float>, "in"_a.noconvert());
		
	m.def("rfft",&rFFT_py<double,int>, "in"_a.noconvert(),"N"_a);
	m.def("rfft",&rFFT_py<float,int>, "in"_a.noconvert(),"N"_a);
	m.def("rfft_train",&rFFT_train_py<double,int>, "in"_a.noconvert(),"N"_a);
	m.def("rfft_train",&rFFT_train_py<float,int>, "in"_a.noconvert(),"N"_a);
	
	//m.def("rfft_parallel",&rFFT_Block_Parallel_py<double>, "in"_a.noconvert(),"N"_a,"nthreads"_a);
	//m.def("rfft_parallel",&rFFT_Block_Parallel_py<float>, "in"_a.noconvert(),"N"_a,"nthreads"_a);
	
	m.def("irfft",&irFFT_py<double>, "in"_a.noconvert());
	m.def("irfft",&irFFT_py<float>, "in"_a.noconvert());
	m.def("irfft_train",&irFFT_train_py<double>, "in"_a.noconvert());
	m.def("irfft_train",&irFFT_train_py<float>, "in"_a.noconvert());
	
	/*m.def("digitizer_FFT",&digitizer_FFT_py<uint8_t>,"in"_a.noconvert(),"conv"_a,"offset"_a);
	m.def("digitizer_FFT",&digitizer_FFT_py<uint16_t>,"in"_a.noconvert(),"conv"_a,"offset"_a);
	m.def("digitizer_FFT",&digitizer_FFT_py<int16_t>,"in"_a.noconvert(),"conv"_a,"offset"_a);
	m.def("digitizer_rFFT",&digitizer_rFFT_py<uint8_t>,"in"_a.noconvert(),"conv"_a,"offset"_a);
	m.def("digitizer_rFFT",&digitizer_rFFT_py<uint16_t>,"in"_a.noconvert(),"conv"_a,"offset"_a);
	m.def("digitizer_rFFT",&digitizer_rFFT_py<int16_t>,"in"_a.noconvert(),"conv"_a,"offset"_a);
	m.def("digitizer_FFT",&digitizer_FFT_Block_py<uint8_t>,"in"_a.noconvert(),"N"_a,"conv"_a,"offset"_a);
	m.def("digitizer_FFT",&digitizer_FFT_Block_py<uint16_t>,"in"_a.noconvert(),"N"_a,"conv"_a,"offset"_a);
	m.def("digitizer_FFT",&digitizer_FFT_Block_py<int16_t>,"in"_a.noconvert(),"N"_a,"conv"_a,"offset"_a);
	m.def("digitizer_rFFT",&digitizer_rFFT_Block_py<uint8_t>,"in"_a.noconvert(),"N"_a,"conv"_a,"offset"_a);
	m.def("digitizer_rFFT",&digitizer_rFFT_Block_py<uint16_t>,"in"_a.noconvert(),"N"_a,"conv"_a,"offset"_a);
	m.def("digitizer_rFFT",&digitizer_rFFT_Block_py<int16_t>,"in"_a.noconvert(),"N"_a,"conv"_a,"offset"_a);
	*/

	fftw_import_wisdom_from_filename(&wisdom_path[0]);
	fftwf_import_wisdom_from_filename(&wisdom_path[0]);
	//fftw_import_wisdom_from_filename(&wisdom_parallel_path[0]);

}

PYBIND11_MODULE(libfft, m)
{
	m.doc() = "Might work, might not.";
	init_fft(m);
}
