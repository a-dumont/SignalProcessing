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
}

PYBIND11_MODULE(libcorrelationscuda, m)
{
	m.doc() = "Contains correlation methods using FFTs";
	init_correlations(m);
}
