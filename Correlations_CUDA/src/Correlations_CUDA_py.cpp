#include "Correlations_CUDA_py.h"
#include <complex>
#include <type_traits>

void init_correlations(py::module &m)
{
	m.def("auto_correlation_cuda",&autocorrelation_cuda_py<np_double>, "In"_a, "Given only 1 argument, will compute the full length auto-correlation");
	m.def("cross_correlation_cuda",&xcorrelation_cuda_py<np_double>, "In1"_a, "In2"_a,"Given only 2 arguments, will compute the full length cross-correlation");
	m.def("cross_correlation_cuda",&xcorrelation_block_cuda_py<np_double>, "In1"_a, "In2"_a,"size"_a,"Given only 2 arguments, will compute the full length cross-correlation");
}

PYBIND11_MODULE(libcorrelationscuda, m)
{
	m.doc() = "Contains correlation methods using FFTs";
	init_correlations(m);
}
