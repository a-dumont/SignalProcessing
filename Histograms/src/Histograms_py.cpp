#include "Histograms_py.h"

void init_histograms(py::module &m)
{
	m.def("histogram",&Histogram_py<np_double,int>, "x"_a,"bins"_a,"A function that generates an histogram based on the number of bins.");
	m.def("histogram",&Histogram_Density_py<np_double,bool>, "x"_a,"bins"_a,"density"_a,"A function that generates an histogram based on the number of bins.");
	m.def("histogram",&Histogram_py<np_double>, "x"_a,"bins"_a,"A function that generates an histogram based on pre determined bins.");
	m.def("histogram",&Histogram_Density_py<np_double>, "x"_a,"bins"_a,"density"_a,"A function that generates an histogram based on pre determined bins.");
	m.def("histogram2d",&Histogram_2D_py<np_double>, "x"_a,"y"_a,"bins"_a);
	m.def("histogram2d",&Histogram_2D_py<np_double,np_double>, "x"_a,"y"_a,"bins"_a);
	m.def("histogram2d",&Histogram_2D_Density_py<np_double>,"x"_a,"y"_a,"bins"_a,"density"_a);
}

PYBIND11_MODULE(libhistograms, m)
{
	m.doc() = "Computes Histograms and is meant to be a replacement for numpy in some simple scenarios.";
	init_histograms(m);
}
