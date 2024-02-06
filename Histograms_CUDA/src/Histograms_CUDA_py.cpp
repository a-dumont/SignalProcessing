#include "Histograms_CUDA_py.h"
#include <type_traits>

void init_histograms(py::module &m)
{
	// 1D histograms
	m.def("digitizer_histogram_CUDA",&digitizer_histogram_1d_py<uint8_t>,"data"_a.noconvert());
	m.def("digitizer_histogram_CUDA",&digitizer_histogram_1d_py<uint16_t>,"data"_a.noconvert());
	m.def("digitizer_histogram_CUDA",&digitizer_histogram_1d_py<uint32_t>,"data"_a.noconvert());
	m.def("digitizer_histogram_CUDA",&digitizer_histogram_1d_py<int8_t>,"data"_a.noconvert());
	m.def("digitizer_histogram_CUDA",&digitizer_histogram_1d_py<int16_t>,"data"_a.noconvert());
	m.def("digitizer_histogram_CUDA",&digitizer_histogram_1d_py<int32_t>,"data"_a.noconvert());
	
	// 1D histograms sub-byte
	m.def("digitizer_histogram_CUDA",&digitizer_histogram_subbyte_1d_py<uint8_t>,
					"data"_a.noconvert(),"nbits"_a);


	// 1D histograms step
	m.def("digitizer_histogram_step_CUDA",&digitizer_histogram_step_1d_py<uint8_t>,
					"data"_a.noconvert(),"nbits"_a);

	// 1D histograms 10bits
	m.def("digitizer_histogram_10bits_CUDA",&digitizer_histogram_10bits_1d_py<uint16_t>,
					"data"_a.noconvert());
	m.def("digitizer_histogram_10bits_CUDA",&digitizer_histogram_10bits_1d_py<int16_t>,
					"data"_a.noconvert());
	
	// 2D histograms
	m.def("digitizer_histogram2D_CUDA",&digitizer_histogram_2d_py<uint8_t>,
					"data_x"_a.noconvert(),"data_y"_a.noconvert());
	m.def("digitizer_histogram2D_CUDA",&digitizer_histogram_2d_py<uint16_t>,
					"data_x"_a.noconvert(),"data_y"_a.noconvert());
	m.def("digitizer_histogram2D_CUDA",&digitizer_histogram_2d_py<int8_t>,
					"data_x"_a.noconvert(),"data_y"_a.noconvert());
	m.def("digitizer_histogram2D_CUDA",&digitizer_histogram_2d_py<int16_t>,
					"data_x"_a.noconvert(),"data_y"_a.noconvert());

	// 2D histograms sub-byte
	m.def("digitizer_histogram2D_CUDA",&digitizer_histogram_subbyte_2d_py<uint8_t>,
					"data_x"_a.noconvert(),"data_y"_a.noconvert(),"nbits"_a);

	// 2D histograms step
	m.def("digitizer_histogram2D_step_CUDA",&digitizer_histogram_step_2d_py<uint8_t>,
					"data_x"_a.noconvert(),"data_y"_a.noconvert(),"nbits"_a);

	// 2D histograms 10bits
	m.def("digitizer_histogram2D_10bits_CUDA",&digitizer_histogram_10bits_2d_py<uint16_t>,
					"data_x"_a.noconvert(),"data_y"_a.noconvert());
	m.def("digitizer_histogram2D_10bits_CUDA",&digitizer_histogram_10bits_2d_py<int16_t>,
					"data_x"_a.noconvert(),"data_y"_a.noconvert());

	// Class
	py::class_<Digitizer_histogram2D_step_CUDA_py>(m,"Digitizer_Step2D_mixed")
		.def(py::init<uint64_t,uint64_t>())
		.def("getHistogram",&Digitizer_histogram2D_step_CUDA_py::getHistogram)
		.def("getCount",&Digitizer_histogram2D_step_CUDA_py::getCount)
		.def("getNbits",&Digitizer_histogram2D_step_CUDA_py::getNbits)
		.def("getSize",&Digitizer_histogram2D_step_CUDA_py::getSize)
		.def("resetHistogram",&Digitizer_histogram2D_step_CUDA_py::resetHistogram)
		.def("accumulate",&Digitizer_histogram2D_step_CUDA_py::accumulate)
		.def("getThreads",&Digitizer_histogram2D_step_CUDA_py::getThreads);

}

PYBIND11_MODULE(libhistogramscuda, m)
{
	m.doc() = "Histograms computed using CUDA";
	init_histograms(m);
}
