#include "Correlations_CUDA_py.h"
#include <type_traits>

void init_correlations(py::module &m)
{
	m.def("auto_correlation_cuda",&autocorrelation_cuda_py<double>, "In"_a.noconvert());
	m.def("auto_correlation_cuda",&autocorrelation_cuda_py<float>, "In"_a.noconvert());
	m.def("auto_correlation_cuda",&autocorrelation_block_cuda_py<double>, "In"_a.noconvert(), 
					"size"_a);
	m.def("auto_correlation_cuda",&autocorrelation_block_cuda_py<float>, "In"_a.noconvert(), 
					"size"_a);

	m.def("cross_correlation_cuda",&cross_correlation_cuda_py<double>, "In1"_a.noconvert(),
					"In2"_a.noconvert());
	m.def("cross_correlation_cuda",&cross_correlation_cuda_py<float>, "In1"_a.noconvert(),
					"In2"_a.noconvert());
	m.def("cross_correlation_cuda",&cross_correlation_block_cuda_py<double>, "In1"_a.noconvert(),
					"In2"_a.noconvert(),"size"_a);
	m.def("cross_correlation_cuda",&cross_correlation_block_cuda_py<float>, "In1"_a.noconvert(),
					"In2"_a.noconvert(),"size"_a);

	m.def("complete_correlation_cuda",&complete_correlation_cuda_py<double>, "In1"_a.noconvert(),
					"In2"_a.noconvert());
	m.def("complete_correlation_cuda",&complete_correlation_cuda_py<float>, "In1"_a.noconvert(),
					"In2"_a.noconvert());
	m.def("complete_correlation_cuda",&complete_correlation_block_cuda_py<double>, 
					"In1"_a.noconvert(),"In2"_a.noconvert(),"size"_a);
	m.def("complete_correlation_cuda",&complete_correlation_block_cuda_py<float>, 
					"In1"_a.noconvert(),"In2"_a.noconvert(),"size"_a);

	m.def("digitizer_autocorrelation_cuda",&digitizer_autocorrelation_cuda_py<uint8_t,double>,
					"In"_a.noconvert(),"size"_a,"conv"_a,"offset"_a);
	m.def("digitizer_autocorrelation_cuda",&digitizer_autocorrelation_cuda_py<uint16_t,double>,
					"In"_a.noconvert(),"size"_a,"conv"_a,"offset"_a);
	m.def("digitizer_autocorrelation_cudaf",&digitizer_autocorrelation_cuda_py<uint8_t,float>,
					"In"_a.noconvert(),"size"_a,"conv"_a,"offset"_a);
	m.def("digitizer_autocorrelation_cudaf",&digitizer_autocorrelation_cuda_py<uint16_t,float>,
					"In"_a.noconvert(),"size"_a,"conv"_a,"offset"_a);

	m.def("digitizer_crosscorrelation_cuda",&digitizer_crosscorrelation_cuda_py<uint8_t,double>,
					"In1"_a.noconvert(),"In2"_a.noconvert(),"size"_a,"conv"_a,"offset"_a);
	m.def("digitizer_crosscorrelation_cuda",&digitizer_crosscorrelation_cuda_py<uint16_t,double>,
					"In1"_a.noconvert(),"In2"_a.noconvert(),"size"_a,"conv"_a,"offset"_a);
	m.def("digitizer_crosscorrelation_cudaf",&digitizer_crosscorrelation_cuda_py<uint8_t,float>,
					"In1"_a.noconvert(),"In2"_a.noconvert(),"size"_a,"conv"_a,"offset"_a);
	m.def("digitizer_crosscorrelation_cudaf",&digitizer_crosscorrelation_cuda_py<uint16_t,float>,
					"In1"_a.noconvert(),"In2"_a.noconvert(),"size"_a,"conv"_a,"offset"_a);

	m.def("digitizer_completecorrelation_cuda",
					&digitizer_completecorrelation_cuda_py<uint8_t,double>,
					"In1"_a.noconvert(),"In2"_a.noconvert(),"size"_a,"conv"_a,"offset"_a);
	m.def("digitizer_completecorrelation_cuda",
					&digitizer_completecorrelation_cuda_py<uint16_t,double>,
					"In1"_a.noconvert(),"In2"_a.noconvert(),"size"_a,"conv"_a,"offset"_a);
	m.def("digitizer_completecorrelation_cudaf",
					&digitizer_completecorrelation_cuda_py<uint8_t,float>,
					"In1"_a.noconvert(),"In2"_a.noconvert(),"size"_a,"conv"_a,"offset"_a);
	m.def("digitizer_completecorrelation_cudaf",
					&digitizer_completecorrelation_cuda_py<uint16_t,float>,
					"In1"_a.noconvert(),"In2"_a.noconvert(),"size"_a,"conv"_a,"offset"_a);

	py::class_<DigitizerAutoCorrelationCuda<uint8_t>>(m,"DigitizerAutoCorrelationCuda")
			.def(py::init<llint_t,llint_t,float,llint_t>())
			.def("accumulate",&DigitizerAutoCorrelationCuda<uint8_t>::accumulate)
			.def("clear",&DigitizerAutoCorrelationCuda<uint8_t>::clear)
			.def("getResult",&DigitizerAutoCorrelationCuda<uint8_t>::getResult);

	py::class_<DigitizerAutoCorrelationCuda<uint16_t>>(m,"DigitizerAutoCorrelationCuda16")
			.def(py::init<llint_t,llint_t,float,llint_t>())
			.def("accumulate",&DigitizerAutoCorrelationCuda<uint16_t>::accumulate)
			.def("clear",&DigitizerAutoCorrelationCuda<uint16_t>::clear)
			.def("getResult",&DigitizerAutoCorrelationCuda<uint16_t>::getResult);

	py::class_<DigitizerCrossCorrelationCuda<uint8_t>>(m,"DigitizerCrossCorrelationCuda")
			.def(py::init<llint_t,llint_t,float,llint_t>())
			.def("accumulate",&DigitizerCrossCorrelationCuda<uint8_t>::accumulate)
			.def("clear",&DigitizerCrossCorrelationCuda<uint8_t>::clear)
			.def("getResult",&DigitizerCrossCorrelationCuda<uint8_t>::getResult);
}

PYBIND11_MODULE(libcorrelationscuda, m)
{
	m.doc() = "Contains correlation methods using FFTs";
	init_correlations(m);
}
