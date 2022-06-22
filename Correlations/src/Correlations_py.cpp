#include "Correlations_py.h"

void init_correlations(py::module &m)
{
	m.def("auto_correlation",&autocorrelation_py<double>, "In"_a, "Given only 1 argument, will compute the full length auto-correlation");
	m.def("auto_correlation",&autocorrelation_py<double,int>, "In"_a, "N"_a, "Will compute the length N auto-correlation");
	m.def("cross_correlation",&xcorrelation_py<double>, "In1"_a, "In2"_a,"Will compute the full length cross-correlation");
	m.def("cross_correlation",&xcorrelation_py<double,int>, "In1"_a, "In2"_a, "N"_a, "Will compute the length N cross-correlation");
	m.def("complete_correlation",&complete_correlation_py<double>, "In1"_a, "In2"_a,"Will compute all 3 full length correlations");
	m.def("complete_correlation",&complete_correlation_py<double,int>, "In1"_a, "In2"_a, "N"_a, "Will compute all 3 length N correlations");
}

PYBIND11_MODULE(libcorrelations, m)
{
	m.doc() = "Contains correlation methods using FFTs";
	init_correlations(m);
}
