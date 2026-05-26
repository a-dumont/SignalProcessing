#include "vkMath_py.h"

void init_module(py::module &m)
{
	// sum
	m.def("vkAdd", &vkAdd<uint32_t>, "computer"_a, "pipeline"_a, "in1"_a.noconvert(), "in2"_a.noconvert());	
}

PYBIND11_MODULE(libvkmath, m)
{
	m.doc() = "Might work, might not.";
	init_module(m);
}
