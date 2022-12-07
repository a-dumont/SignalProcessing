#include "Histograms_CUDA_py.h"

void init_histograms(py::module &m)
{
	m.def("digitizer_histogram_filter",&digitizer_histogram_filter_py<uint8_t>,"data"_a.noconvert(), "filter"_a, "offset"_a);
	m.def("digitizer_histogram_filter",&digitizer_histogram_filter_py<uint16_t>,"data"_a.noconvert(), "filter"_a, "offset"_a);
}

PYBIND11_MODULE(libhistogramscuda, m)
{
	m.doc() = "Computes Histograms and is meant to be a replacement for numpy in some simple scenarios.";
	init_histograms(m);
}
