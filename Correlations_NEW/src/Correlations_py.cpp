#include "Correlations_py.h"

void init_correlations(py::module &m)
{
	m.def("reduceAVX",&reduceAVX_py<float>, "In"_a.noconvert());
	m.def("reduceAVX",&reduceAVX_py<double>, "In"_a.noconvert());
	m.def("uint_to_hex",&base16_py<uint64_t>, "In"_a.noconvert());
	m.def("uint_to_oct",&base8_py<uint64_t>, "In"_a.noconvert());
	m.def("uint_to_128",&base128_py<uint64_t>, "In"_a.noconvert());
}

PYBIND11_MODULE(libcorrelations, m)
{
	m.doc() = "Contains correlation methods using FFTs";
	init_correlations(m);
}
