#include "FFT_py.h"

void init_fft(py::module &m)
{
	// FFT
	m.def("fft",&fft_py<double>,"In"_a.noconvert());
	m.def("fft",&fft_py<float>,"In"_a.noconvert());
	m.def("fft_training",&fft_training_py<double>,"In"_a.noconvert());
	m.def("fft_training",&fft_training_py<float>,"In"_a.noconvert());

	py::class_<FFT_py>(m,"FFT")
			.def(py::init<uint64_t>())
			.def("fft",&FFT_py::fft,"In"_a.noconvert())
			.def("fft",&FFT_py::fftf,"In"_a.noconvert())
			.def("getN",&FFT_py::getN);
	
	// iFFT
	m.def("ifft",&ifft_py<double>,"In"_a.noconvert());
	m.def("ifft",&ifft_py<float>,"In"_a.noconvert());
	m.def("ifft_training",&ifft_training_py<double>,"In"_a.noconvert());
	m.def("ifft_training",&ifft_training_py<float>,"In"_a.noconvert());

	py::class_<IFFT_py>(m,"IFFT")
			.def(py::init<uint64_t>())
			.def("ifft",&IFFT_py::ifft,"In"_a.noconvert())
			.def("ifft",&IFFT_py::ifftf,"In"_a.noconvert())
			.def("getN",&IFFT_py::getN);

	// rFFT
	m.def("rfft",&rfft_py<double>,"In"_a.noconvert());
	m.def("rfft",&rfft_py<float>,"In"_a.noconvert());
	m.def("rfft_training",&rfft_training_py<double>,"In"_a.noconvert());
	m.def("rfft_training",&rfft_training_py<float>,"In"_a.noconvert());

	py::class_<RFFT_py>(m,"RFFT")
			.def(py::init<uint64_t>())
			.def("rfft",&RFFT_py::rfft,"In"_a.noconvert())
			.def("rfft",&RFFT_py::rfftf,"In"_a.noconvert())
			.def("getN",&RFFT_py::getN);

	// irFFT
	m.def("irfft",&irfft_py<double>,"In"_a.noconvert());
	m.def("irfft",&irfft_py<float>,"In"_a.noconvert());
	m.def("irfft_training",&irfft_training_py<double>,"In"_a.noconvert());
	m.def("irfft_training",&irfft_training_py<float>,"In"_a.noconvert());

	py::class_<IRFFT_py>(m,"IRFFT")
			.def(py::init<uint64_t>())
			.def("irfft",&IRFFT_py::irfft,"In"_a.noconvert())
			.def("irfft",&IRFFT_py::irfftf,"In"_a.noconvert())
			.def("getN",&IRFFT_py::getN);


	// Load wisdom
	fftw_import_wisdom_from_filename(&wisdom_path[0]);
	//fftwf_import_wisdom_from_filename(&wisdom_path[0]);
}

PYBIND11_MODULE(libfft, m)
{
	m.doc() = "Might work, might not.";
	init_fft(m);
}
