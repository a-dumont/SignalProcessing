#include "Math_py.h"

void init_module(py::module &m)
{
	m.def("gradient", &gradient_py<np_double>, "x"_a, "t"_a);
	m.def("gradient", &gradient_py<np_double,double>, "x"_a, "dt"_a);
	m.def("gradient", &gradient_py<np_double,int>, "x"_a, "dt"_a);
	m.def("rolling_average",&rolling_average_py<np_double>, "in"_a,"size"_a);
	m.def("finite_difference_coefficients",&finite_difference_coefficients_py<np_double>, "M"_a,"N"_a);
	m.def("nth_order_gradient",&nth_order_gradient_py<np_double,double>,"x"_a,"dt"_a,"M"_a,"N"_a);
	m.def("continuous_max",&continuous_max_py<double>,"in"_a);
	m.def("continuous_min",&continuous_min_py<double>,"in"_a);
	m.def("Sum",&sum_py<int>,"in"_a);
	m.def("Sum",&sum_py<long int>,"in"_a);
	m.def("Sum",&sum_py<double>,"in"_a);
	m.def("Sum",&sum_py<dbl_complex>,"in"_a);
	m.def("mean",&mean_py<int>,"in"_a);
	m.def("mean",&mean_py<double>,"in"_a);
	m.def("mean",&mean_complex_py<dbl_complex>,"in"_a);
	m.def("variance",&variance_py<double>,"in"_a);
	m.def("skewness",&skewness_py<double>,"in"_a);
	m.def("product",&product_py<int,int>,"in1"_a,"in2"_a);
	m.def("product",&product_py<double,double>,"in1"_a,"in2"_a);
	m.def("product",&product_py<dbl_complex,dbl_complex>,"in1"_a,"in2"_a);
	m.def("Sum",&sum_py<int,int>,"in1"_a,"in2"_a);
	m.def("Sum",&sum_py<double,double>,"in1"_a,"in2"_a);
	m.def("Sum",&sum_py<dbl_complex,dbl_complex>,"in1"_a,"in2"_a);
	m.def("difference",&difference_py<int,int>,"in1"_a,"in2"_a);
	m.def("difference",&difference_py<double,double>,"in1"_a,"in2"_a);
	m.def("difference",&difference_py<dbl_complex,dbl_complex>,"in1"_a,"in2"_a);
	m.def("division",&division_py<int,int>,"in1"_a,"in2"_a);
	m.def("division",&division_py<double,double>,"in1"_a,"in2"_a);
	m.def("division",&division_py<dbl_complex,dbl_complex>,"in1"_a,"in2"_a);
	m.def("minimum",&min_py<int>,"in"_a);
	m.def("minimum",&min_py<double>,"in"_a);
	m.def("maximum",&max_py<int>,"in"_a);
	m.def("maximum",&max_py<double>,"in"_a);
}

PYBIND11_MODULE(libmath, m)
{
	m.doc() = "Might work, might not.";
	init_module(m);
}
