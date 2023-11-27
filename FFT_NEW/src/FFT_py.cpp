#include "FFT_py.h"

void init_fft(py::module &m)
{
	// FFT
	m.def("fft",&fft_py<double>,"In"_a.noconvert());
	m.def("fft",&fft_py<float>,"In"_a.noconvert());
	m.def("fft",&fft_pad_py<double>,"In"_a.noconvert(),"size"_a);
	m.def("fft",&fft_pad_py<float>,"In"_a.noconvert(),"size"_a);
	m.def("fft_training",&fft_training_py<double>,"In"_a.noconvert());
	m.def("fft_training",&fft_training_py<float>,"In"_a.noconvert());
	m.def("fftBlock",&fftBlock_py<double>,"In"_a.noconvert(),"size"_a);
	m.def("fftBlock",&fftBlock_py<float>,"In"_a.noconvert(),"size"_a);
	m.def("fftBlock_training",&fftBlock_training_py<double>,"In"_a.noconvert(),"size"_a);
	m.def("fftBlock_training",&fftBlock_training_py<float>,"In"_a.noconvert(),"size"_a);

	py::class_<FFT_py>(m,"FFT")
			.def(py::init<uint64_t>())
			.def("fft",&FFT_py::fft,"In"_a.noconvert())
			.def("fft",&FFT_py::fftf,"In"_a.noconvert())
			.def("getSize",&FFT_py::getSize)
			.def("benchmark",&FFT_py::benchmark)
			.def("train",&FFT_py::train);

	py::class_<FFT_Block_py>(m,"FFT_Block")
			.def(py::init<uint64_t,uint64_t>())
			.def("fftBlock",&FFT_Block_py::fftBlock,"In"_a.noconvert())
			.def("fftBlock",&FFT_Block_py::fftBlockf,"In"_a.noconvert())
			.def("getSize",&FFT_Block_py::getSize)
			.def("getN",&FFT_Block_py::getN)
			.def("getHowmany",&FFT_Block_py::getHowmany)
			.def("benchmark",&FFT_Block_py::benchmark)
			.def("train",&FFT_Block_py::train);
	
	// iFFT
	m.def("ifft",&ifft_py<double>,"In"_a.noconvert());
	m.def("ifft",&ifft_py<float>,"In"_a.noconvert());
	m.def("ifft",&ifft_pad_py<double>,"In"_a.noconvert(),"size"_a);
	m.def("ifft",&ifft_pad_py<float>,"In"_a.noconvert(),"size"_a);
	m.def("ifft_training",&ifft_training_py<double>,"In"_a.noconvert());
	m.def("ifft_training",&ifft_training_py<float>,"In"_a.noconvert());
	m.def("ifftBlock",&ifftBlock_py<double>,"In"_a.noconvert(),"size"_a);
	m.def("ifftBlock",&ifftBlock_py<float>,"In"_a.noconvert(),"size"_a);
	m.def("ifftBlock_training",&ifftBlock_training_py<double>,"In"_a.noconvert(),"size"_a);
	m.def("ifftBlock_training",&ifftBlock_training_py<float>,"In"_a.noconvert(),"size"_a);

	py::class_<IFFT_py>(m,"IFFT")
			.def(py::init<uint64_t>())
			.def("ifft",&IFFT_py::ifft,"In"_a.noconvert())
			.def("ifft",&IFFT_py::ifftf,"In"_a.noconvert())
			.def("getSize",&IFFT_py::getSize)
			.def("benchmark",&IFFT_py::benchmark)
			.def("train",&IFFT_py::train);

	py::class_<IFFT_Block_py>(m,"IFFT_Block")
			.def(py::init<uint64_t,uint64_t>())
			.def("ifftBlock",&IFFT_Block_py::ifftBlock,"In"_a.noconvert())
			.def("ifftBlock",&IFFT_Block_py::ifftBlockf,"In"_a.noconvert())
			.def("getSize",&IFFT_Block_py::getSize)
			.def("getN",&IFFT_Block_py::getN)
			.def("getHowmany",&IFFT_Block_py::getHowmany)
			.def("benchmark",&IFFT_Block_py::benchmark)
			.def("train",&IFFT_Block_py::train);

	// rFFT
	m.def("rfft",&rfft_py<double>,"In"_a.noconvert());
	m.def("rfft",&rfft_py<float>,"In"_a.noconvert());
	m.def("rfft",&rfft_pad_py<double>,"In"_a.noconvert(),"size"_a);
	m.def("rfft",&rfft_pad_py<float>,"In"_a.noconvert(),"size"_a);
	m.def("rfft_training",&rfft_training_py<double>,"In"_a.noconvert());
	m.def("rfft_training",&rfft_training_py<float>,"In"_a.noconvert());
	m.def("rfftBlock",&rfftBlock_py<double>,"In"_a.noconvert(),"size"_a);
	m.def("rfftBlock",&rfftBlock_py<float>,"In"_a.noconvert(),"size"_a);
	m.def("rfftBlock_training",&rfftBlock_training_py<double>,"In"_a.noconvert(),"size"_a);
	m.def("rfftBlock_training",&rfftBlock_training_py<float>,"In"_a.noconvert(),"size"_a);
	m.def("digitizer_rfft",&digitizer_rfft_py<uint8_t,double>,
					"In"_a.noconvert(),"conv"_a,"offset"_a);
	m.def("digitizer_rfft",&digitizer_rfft_py<int16_t,double>,
					"In"_a.noconvert(),"conv"_a,"offset"_a);
	m.def("digitizer_rfftf",&digitizer_rfft_py<uint8_t, float>,
					"In"_a.noconvert(),"conv"_a,"offset"_a);
	m.def("digitizer_rfftf",&digitizer_rfft_py<int16_t, float>,
					"In"_a.noconvert(),"conv"_a,"offset"_a);


	py::class_<RFFT_py>(m,"RFFT")
			.def(py::init<uint64_t>())
			.def("rfft",&RFFT_py::rfft,"In"_a.noconvert())
			.def("rfft",&RFFT_py::rfftf,"In"_a.noconvert())
			.def("getSize",&RFFT_py::getSize)
			.def("benchmark",&RFFT_py::benchmark)
			.def("train",&RFFT_py::train);


	py::class_<RFFT_Block_py>(m,"RFFT_Block")
			.def(py::init<uint64_t,uint64_t>())
			.def("rfftBlock",&RFFT_Block_py::rfftBlock,"In"_a.noconvert())
			.def("rfftBlock",&RFFT_Block_py::rfftBlockf,"In"_a.noconvert())
			.def("getSize",&RFFT_Block_py::getSize)
			.def("getN",&RFFT_Block_py::getN)
			.def("getHowmany",&RFFT_Block_py::getHowmany)
			.def("benchmark",&RFFT_Block_py::benchmark)
			.def("train",&RFFT_Block_py::train);

	// irFFT
	m.def("irfft",&irfft_py<double>,"In"_a.noconvert());
	m.def("irfft",&irfft_py<float>,"In"_a.noconvert());
	m.def("irfft",&irfft_pad_py<double>,"In"_a.noconvert(),"size"_a);
	m.def("irfft",&irfft_pad_py<float>,"In"_a.noconvert(),"size"_a);
	m.def("irfft_training",&irfft_training_py<double>,"In"_a.noconvert());
	m.def("irfft_training",&irfft_training_py<float>,"In"_a.noconvert());
	m.def("irfftBlock",&irfftBlock_py<double>,"In"_a.noconvert(),"size"_a);
	m.def("irfftBlock",&irfftBlock_py<float>,"In"_a.noconvert(),"size"_a);

	py::class_<IRFFT_py>(m,"IRFFT")
			.def(py::init<uint64_t>())
			.def("irfft",&IRFFT_py::irfft,"In"_a.noconvert())
			.def("irfft",&IRFFT_py::irfftf,"In"_a.noconvert())
			.def("getSize",&IRFFT_py::getSize)
			.def("benchmark",&IRFFT_py::benchmark)
			.def("train",&IRFFT_py::train);

	py::class_<IRFFT_Block_py>(m,"IRFFT_Block")
			.def(py::init<uint64_t,uint64_t>())
			.def("irfftBlock",&IRFFT_Block_py::irfftBlock,"In"_a.noconvert())
			.def("irfftBlock",&IRFFT_Block_py::irfftBlockf,"In"_a.noconvert())
			.def("getSize",&IRFFT_Block_py::getSize)
			.def("getN",&IRFFT_Block_py::getN)
			.def("getHowmany",&IRFFT_Block_py::getHowmany)
			.def("benchmark",&IRFFT_Block_py::benchmark)
			.def("train",&IRFFT_Block_py::train);

	// Others
	m.def("convert",&convertAVX_py<uint8_t,double>,"In"_a.noconvert(),"conv"_a,"offset"_a);
	m.def("convert",&convertAVX_py<int16_t,double>,"In"_a.noconvert(),"conv"_a,"offset"_a);
	m.def("convertf",&convertAVX_py<uint8_t,float>,"In"_a.noconvert(),"conv"_a,"offset"_a);
	m.def("convertf",&convertAVX_py<int16_t,float>,"In"_a.noconvert(),"conv"_a,"offset"_a);

	// Load wisdom
	fftw_import_wisdom_from_filename(&wisdom_path[0]);
	//fftwf_import_wisdom_from_filename(&wisdom_path[0]);
	
	// Set max plan time in seconds
	fftw_set_timelimit(3000);	
}

PYBIND11_MODULE(libfft, m)
{
	m.doc() = "Might work, might not.";
	init_fft(m);
}
