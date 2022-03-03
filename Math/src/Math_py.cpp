#include "Math_py.h"

void init_module(py::module &m)
{
	m.def("gradient", &gradient_py<np_double>, "x"_a, "t"_a);
	m.def("gradient", &gradient_py<np_double,double>, "x"_a, "dt"_a);
	m.def("rolling_average",&rolling_average_py<np_double>, "in"_a,"size"_a);
	m.def("finite_difference_coefficients",&finite_difference_coefficients_py<np_double>, "M"_a,"N"_a);
	m.def("nth_order_gradient",&nth_order_gradient_py<np_double,double>,"x"_a,"dt"_a,"M"_a,"N"_a);
	m.def("histogram_vectorial_average",&histogram_vectorial_average_py<np_double>,"hist"_a,"row"_a,"col"_a);
	m.def("inverse_probability2D",&inverse_probability2D_py<np_double>,"probabilities"_a,"probability_density"_a);
}

PYBIND11_MODULE(libmath, m)
{
	m.doc() = "Might work, might not.";
	init_module(m);
}
