#include "Math_CUDA_py.h"

void init_math(py::module &m)
{
	m.def("vAdd",&vector_sum_py<DataType>, "In1"_a.noconvert(),"In2"_a.noconvert());
}

PYBIND11_MODULE(libmathcuda, m)
{
	m.doc() = "Useful math using CUDA";
	init_math(m);
}