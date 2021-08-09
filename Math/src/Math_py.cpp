#include "Math_py.h"

void init_module(py::module &m)
{
	m.def("gradient", &gradient_py<np_double>, "x"_a, "t"_a);
	m.def("gradient", &gradient_py<np_double,double>, "x"_a, "dt"_a);
}

PYBIND11_MODULE(libmath, m)
{
	m.doc() = "Might work, might not.";
	init_module(m);
}
