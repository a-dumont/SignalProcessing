#include "vkMath_py.h"

void init_module(py::module &m)
{
	// sum
	m.def("vkAddBuilder", &vkAddBuilder<uint32_t>, "computer"_a, "pipeline"_a);	
}

PYBIND11_MODULE(libvkmath, m)
{
	m.doc() = "Might work, might not.";
	init_module(m);
}
