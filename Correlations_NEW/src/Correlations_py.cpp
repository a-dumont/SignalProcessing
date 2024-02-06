#include "Correlations_py.h"

void init_correlations(py::module &m)
{
	// Acorr
	m.def("aCorrCircularFreqAVX",&aCorrCircularFreqAVX_py<float>, "In"_a.noconvert(),"size"_a);
	m.def("aCorrCircularFreqAVX",&aCorrCircularFreqAVX_py<double>, "In"_a.noconvert(),"size"_a);
	
	py::class_<ACorrCircularFreqAVX_py>(m,"ACorrCircularFreqAVX")
			.def(py::init<uint64_t,uint64_t>())
			.def("aCorrCircularFreqAVX",&ACorrCircularFreqAVX_py::aCorrCircularFreqAVX,
				"In"_a.noconvert())
			.def("aCorrCircularFreqAVX",&ACorrCircularFreqAVX_py::aCorrCircularFreqAVXf,
				"In"_a.noconvert())
			.def("getSize",&ACorrCircularFreqAVX_py::getSize)
			.def("getN",&ACorrCircularFreqAVX_py::getN)
			.def("getHowmany",&ACorrCircularFreqAVX_py::getHowmany)
			.def("benchmark",&ACorrCircularFreqAVX_py::benchmark)
			.def("train",&ACorrCircularFreqAVX_py::train);

	py::class_<DigitizerACorrCircularFreqAVX_py>(m,"DigitizerACorrCircularFreqAVX")
			.def(py::init<uint64_t,uint64_t>())
			.def("aCorrCircularFreqAVX",
				&DigitizerACorrCircularFreqAVX_py::aCorrCircularFreqAVX<uint8_t>,
				"In"_a.noconvert(),"Conv"_a,"offset"_a)
			.def("aCorrCircularFreqAVX",
				&DigitizerACorrCircularFreqAVX_py::aCorrCircularFreqAVX<int16_t>,
				"In"_a.noconvert(),"Conv"_a,"offset"_a)
			.def("aCorrCircularFreqAVXf",
				&DigitizerACorrCircularFreqAVX_py::aCorrCircularFreqAVXf<uint8_t>,
				"In"_a.noconvert(),"Conv"_a,"offset"_a)
			.def("aCorrCircularFreqAVXf",
				&DigitizerACorrCircularFreqAVX_py::aCorrCircularFreqAVXf<int16_t>,
				"In"_a.noconvert(),"Conv"_a,"offset"_a)
			.def("getSize",&DigitizerACorrCircularFreqAVX_py::getSize)
			.def("getN",&DigitizerACorrCircularFreqAVX_py::getN)
			.def("getHowmany",&DigitizerACorrCircularFreqAVX_py::getHowmany)
			.def("benchmark",&DigitizerACorrCircularFreqAVX_py::benchmark)
			.def("train",&DigitizerACorrCircularFreqAVX_py::train);

	
	// Xcorr
	m.def("xCorrCircularFreqAVX",&xCorrCircularFreqAVX_py<float>, 
					"In1"_a.noconvert(), "In2"_a.noconvert(), "size"_a);
	m.def("xCorrCircularFreqAVX",&xCorrCircularFreqAVX_py<double>, 
					"In1"_a.noconvert(), "In2"_a.noconvert(), "size"_a);

	py::class_<XCorrCircularFreqAVX_py>(m,"XCorrCircularFreqAVX")
			.def(py::init<uint64_t,uint64_t>())
			.def("xCorrCircularFreqAVX",&XCorrCircularFreqAVX_py::xCorrCircularFreqAVX,
				"In1"_a.noconvert(),"In2"_a.noconvert())
			.def("xCorrCircularFreqAVX",&XCorrCircularFreqAVX_py::xCorrCircularFreqAVXf,
				"In1"_a.noconvert(),"In2"_a.noconvert())
			.def("getSize",&XCorrCircularFreqAVX_py::getSize)
			.def("getN",&XCorrCircularFreqAVX_py::getN)
			.def("getHowmany",&XCorrCircularFreqAVX_py::getHowmany)
			.def("benchmark",&XCorrCircularFreqAVX_py::benchmark)
			.def("train",&XCorrCircularFreqAVX_py::train);

	py::class_<DigitizerXCorrCircularFreqAVX_py>(m,"DigitizerXCorrCircularFreqAVX")
			.def(py::init<uint64_t,uint64_t>())
			.def("xCorrCircularFreqAVX",
				&DigitizerXCorrCircularFreqAVX_py::xCorrCircularFreqAVX<uint8_t>,
				"In1"_a.noconvert(),"In2"_a.noconvert(),"Conv"_a,"offset"_a)
			.def("xCorrCircularFreqAVX",
				&DigitizerXCorrCircularFreqAVX_py::xCorrCircularFreqAVX<int16_t>,
				"In1"_a.noconvert(),"In2"_a.noconvert(),"Conv"_a,"offset"_a)
			.def("xCorrCircularFreqAVXf",
				&DigitizerXCorrCircularFreqAVX_py::xCorrCircularFreqAVXf<uint8_t>,
				"In1"_a.noconvert(),"In2"_a.noconvert(),"Conv"_a,"offset"_a)
			.def("xCorrCircularFreqAVXf",
				&DigitizerXCorrCircularFreqAVX_py::xCorrCircularFreqAVXf<int16_t>,
				"In1"_a.noconvert(),"In2"_a.noconvert(),"Conv"_a,"offset"_a)
			.def("getSize",&DigitizerXCorrCircularFreqAVX_py::getSize)
			.def("getN",&DigitizerXCorrCircularFreqAVX_py::getN)
			.def("getHowmany",&DigitizerXCorrCircularFreqAVX_py::getHowmany)
			.def("benchmark",&DigitizerXCorrCircularFreqAVX_py::benchmark)
			.def("train",&DigitizerXCorrCircularFreqAVX_py::train);
					
	// Combined Acorr and Xcorr
	m.def("fCorrCircularFreqAVX",&fCorrCircularFreqAVX_py<float>, 
					"In1"_a.noconvert(), "In2"_a.noconvert(), "size"_a);
	m.def("fCorrCircularFreqAVX",&fCorrCircularFreqAVX_py<double>, 
					"In1"_a.noconvert(), "In2"_a.noconvert(), "size"_a);

	py::class_<FCorrCircularFreqAVX_py>(m,"FCorrCircularFreqAVX")
			.def(py::init<uint64_t,uint64_t>())
			.def("fCorrCircularFreqAVX",&FCorrCircularFreqAVX_py::fCorrCircularFreqAVX,
				"In1"_a.noconvert(),"In2"_a.noconvert())
			.def("fCorrCircularFreqAVX",&FCorrCircularFreqAVX_py::fCorrCircularFreqAVXf,
				"In1"_a.noconvert(),"In2"_a.noconvert())
			.def("getSize",&FCorrCircularFreqAVX_py::getSize)
			.def("getN",&FCorrCircularFreqAVX_py::getN)
			.def("getHowmany",&FCorrCircularFreqAVX_py::getHowmany)
			.def("benchmark",&FCorrCircularFreqAVX_py::benchmark)
			.def("train",&FCorrCircularFreqAVX_py::train);


	py::class_<DigitizerFCorrCircularFreqAVX_py>(m,"DigitizerFCorrCircularFreqAVX")
			.def(py::init<uint64_t,uint64_t>())
			.def("fCorrCircularFreqAVX",
				&DigitizerFCorrCircularFreqAVX_py::fCorrCircularFreqAVX<uint8_t>,
				"In1"_a.noconvert(),"In2"_a.noconvert(),"Conv"_a,"offset"_a)
			.def("fCorrCircularFreqAVX",
				&DigitizerFCorrCircularFreqAVX_py::fCorrCircularFreqAVX<int16_t>,
				"In1"_a.noconvert(),"In2"_a.noconvert(),"Conv"_a,"offset"_a)
			.def("fCorrCircularFreqAVXf",
				&DigitizerFCorrCircularFreqAVX_py::fCorrCircularFreqAVXf<uint8_t>,
				"In1"_a.noconvert(),"In2"_a.noconvert(),"Conv"_a,"offset"_a)
			.def("xCorrCircularFreqAVXf",
				&DigitizerFCorrCircularFreqAVX_py::fCorrCircularFreqAVXf<int16_t>,
				"In1"_a.noconvert(),"In2"_a.noconvert(),"Conv"_a,"offset"_a)
			.def("getSize",&DigitizerFCorrCircularFreqAVX_py::getSize)
			.def("getN",&DigitizerFCorrCircularFreqAVX_py::getN)
			.def("getHowmany",&DigitizerFCorrCircularFreqAVX_py::getHowmany)
			.def("benchmark",&DigitizerFCorrCircularFreqAVX_py::benchmark)
			.def("train",&DigitizerFCorrCircularFreqAVX_py::train);	

	// Reduction
	m.def("reduceAVX",&reduceAVX_py<float>, "In"_a.noconvert());
	m.def("reduceAVX",&reduceAVX_py<double>, "In"_a.noconvert());
	
	m.def("reduceAVX",&reduceBlockAVX_py<float>, "In"_a.noconvert(),"size"_a);
	m.def("reduceAVX",&reduceBlockAVX_py<double>, "In"_a.noconvert(),"size"_a);
	
	// Load wisdom
	fftw_import_wisdom_from_filename(&wisdom_path[0]);
	fftwf_import_wisdom_from_filename(&wisdom_pathf[0]);
	
	// Set max plan time in seconds
	fftw_set_timelimit(300);
	fftwf_set_timelimit(300);
}

PYBIND11_MODULE(libcorrelations, m)
{
	m.doc() = "Contains correlation methods.";
	init_correlations(m);
}
