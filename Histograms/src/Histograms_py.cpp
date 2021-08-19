#include "Histograms_py.h"

void init_histograms(py::module &m)
{
	m.def("histogram",&Histogram_py<np_double,int>, "in"_a,"nbins"_a);
	m.def("histogram",&Histogram_py<np_double>, "in"_a,"edges"_a);
}

PYBIND11_MODULE(libhistograms, m)
{
	m.doc() = "Might work, might not.";
	init_histograms(m);
}
