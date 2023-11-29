#include "Correlations_py.h"

void init_correlations(py::module &m)
{
	m.def("reduceAVX",&reduceAVX_py<float>, "In"_a.noconvert());
}

PYBIND11_MODULE(libcorrelations, m)
{
	m.doc() = "Contains correlation methods using FFTs";
	init_correlations(m);
}
