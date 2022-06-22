#include "Math_py.h"

void init_module(py::module &m)
{
	//Fullsize gradient
	m.def("gradient", &gradient_py<double,double>, "x"_a, "t"_a.noconvert());
	m.def("gradient", &gradient_py<double,float>, "x"_a, "t"_a.noconvert());
	m.def("gradient", &gradient_py<double,long long int>, "x"_a, "t"_a.noconvert());
	m.def("gradient", &gradient_py<float,double>, "x"_a, "t"_a.noconvert());
	m.def("gradient", &gradient_py<float,float>, "x"_a, "t"_a.noconvert());
	m.def("gradient", &gradient_py<float,long long int>, "x"_a, "t"_a.noconvert());
	
	//Gradient with constant step
	m.def("gradient", &gradient2_py<double,double>, "x"_a, "dt"_a.noconvert());
	m.def("gradient", &gradient2_py<double,long long int>, "x"_a, "dt"_a.noconvert());
	m.def("gradient", &gradient2_py<float,double>, "x"_a, "dt"_a.noconvert());
	m.def("gradient", &gradient2_py<float,long long int>, "x"_a, "dt"_a.noconvert());

	//Rolling average
	m.def("rolling_average",&rolling_average_py<double>, "in"_a,"size"_a);
	m.def("rolling_average",&rolling_average_py<float>, "in"_a,"size"_a);

	//Finite difference for any reasonable symmetrical stencil
	m.def("finite_difference_coefficients",&finite_difference_coefficients_py<double>,"M"_a,"N"_a);
	m.def("finite_difference_coefficients",&finite_difference_coefficients_py<float>,"M"_a,"N"_a);
	
	m.def("nth_order_gradient",&nth_order_gradient_py<double,double>,"x"_a.noconvert(),"dt"_a,"M"_a,"N"_a);
	m.def("nth_order_gradient",&nth_order_gradient_py<double,float>,"x"_a.noconvert(),"dt"_a,"M"_a,"N"_a);
	m.def("nth_order_gradient",&nth_order_gradient_py<double,long long int>,"x"_a.noconvert(),"dt"_a,"M"_a,"N"_a);
	m.def("nth_order_gradient",&nth_order_gradient_py<float,double>,"x"_a.noconvert(),"dt"_a,"M"_a,"N"_a);
	m.def("nth_order_gradient",&nth_order_gradient_py<float,float>,"x"_a.noconvert(),"dt"_a,"M"_a,"N"_a);
	m.def("nth_order_gradient",&nth_order_gradient_py<float,long long int>,"x"_a.noconvert(),"dt"_a,"M"_a,"N"_a);
	
	//Continuous maximum
	m.def("continuous_max",&continuous_max_py<double>,"in"_a);
	m.def("continuous_max",&continuous_max_py<float>,"in"_a);
	m.def("continuous_max",&continuous_max_py<long long int>,"in"_a);
	
	//Continuous minimum
	m.def("continuous_min",&continuous_min_py<double>,"in"_a);
	m.def("continuous_min",&continuous_min_py<float>,"in"_a);
	m.def("continuous_min",&continuous_min_py<long long int>,"in"_a);
	
	//Reduction of a vector to a scalar
	m.def("Sum",&sum_py<double>,"in"_a);
	m.def("Sum",&sum_py<float>,"in"_a);
	m.def("Sum",&sum_py<long long int>,"in"_a);
	m.def("Sum",&sum_py<dbl_complex>,"in"_a);
	
	//Reduction of a vector into a scalar mean
	m.def("mean",&mean_py<double>,"in"_a);
	m.def("mean",&mean_py<float>,"in"_a);
	m.def("mean",&mean_py<long long int>,"in"_a);
	m.def("mean",&mean_complex_py<dbl_complex>,"in"_a);
	
	//Variance of a vector
	m.def("variance",&variance_py<double>,"in"_a);
	m.def("variance",&variance_py<float>,"in"_a);
	
	//Skewness of a vector
	m.def("skewness",&skewness_py<double>,"in"_a);
	m.def("skewness",&skewness_py<float>,"in"_a);
	
	//Product elementwise of two vectors
	m.def("product",&product_py<double,double>,"in1"_a,"in2"_a);
	m.def("product",&product_py<float,float>,"in1"_a,"in2"_a);
	m.def("product",&product_py<long long int,long long int>,"in1"_a,"in2"_a);
	m.def("product",&product_py<dbl_complex,dbl_complex>,"in1"_a,"in2"_a);
	
	//Sum elementwise of two vectors
	m.def("Sum",&sum_py<double,double>,"in1"_a,"in2"_a);
	m.def("Sum",&sum_py<float,float>,"in1"_a,"in2"_a);
	m.def("Sum",&sum_py<long long int,long long int>,"in1"_a,"in2"_a);
	m.def("Sum",&sum_py<dbl_complex,dbl_complex>,"in1"_a,"in2"_a);
	
	//Difference elementwise of two vectors
	m.def("difference",&difference_py<double,double>,"in1"_a,"in2"_a);
	m.def("difference",&difference_py<float,float>,"in1"_a,"in2"_a);
	m.def("difference",&difference_py<long long int,long long int>,"in1"_a,"in2"_a);
	m.def("difference",&difference_py<dbl_complex,dbl_complex>,"in1"_a,"in2"_a);
	
	//Division elementwise of two vectors
	m.def("division",&division_py<double,double>,"in1"_a,"in2"_a);
	m.def("division",&division_py<float,float>,"in1"_a,"in2"_a);
	m.def("division",&division_py<long long int,long long int>,"in1"_a,"in2"_a);
	m.def("division",&division_py<dbl_complex,dbl_complex>,"in1"_a,"in2"_a);
	
	//Minimum in a vector
	m.def("minimum",&min_py<double>,"in"_a);
	m.def("minimum",&min_py<float>,"in"_a);
	m.def("minimum",&min_py<long long int>,"in"_a);
	
	//Maximum in a vector
	m.def("maximum",&max_py<double>,"in"_a);
	m.def("maximum",&max_py<float>,"in"_a);
	m.def("maximum",&max_py<long long int>,"in"_a);
}

PYBIND11_MODULE(libmath, m)
{
	m.doc() = "Might work, might not.";
	init_module(m);
}
