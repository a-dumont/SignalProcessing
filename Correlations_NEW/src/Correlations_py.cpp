#include "Correlations_py.h"

void init_correlations(py::module &m)
{
	// Acorr
	m.def("aCorrCircularFreqAVX",&aCorrCircularFreqAVX_py<float>, "In"_a.noconvert(),"size"_a);
	m.def("aCorrCircularFreqAVX",&aCorrCircularFreqAVX_py<double>, "In"_a.noconvert(),"size"_a);
	
	// Xcorr
	m.def("xCorrCircularFreqAVX",&xCorrCircularFreqAVX_py<float>, 
					"In1"_a.noconvert(), "In2"_a.noconvert(), "size"_a);
	m.def("xCorrCircularFreqAVX",&xCorrCircularFreqAVX_py<double>, 
					"In1"_a.noconvert(), "In2"_a.noconvert(), "size"_a);
					
	// Combined Acorr and Xcorr
	m.def("axCorrCircularFreqAVX",&axCorrCircularFreqAVX_py<float>, 
					"In1"_a.noconvert(), "In2"_a.noconvert(), "size"_a);
	m.def("axCorrCircularFreqAVX",&axCorrCircularFreqAVX_py<double>, 
					"In1"_a.noconvert(), "In2"_a.noconvert(), "size"_a);
	
	// Reduction
	m.def("reduceAVX",&reduceAVX_py<float>, "In"_a.noconvert());
	m.def("reduceAVX",&reduceAVX_py<double>, "In"_a.noconvert());
	
	m.def("reduceAVX",&reduceBlockAVX_py<float>, "In"_a.noconvert(),"size"_a);
	m.def("reduceAVX",&reduceBlockAVX_py<double>, "In"_a.noconvert(),"size"_a);
}

PYBIND11_MODULE(libcorrelations, m)
{
	m.doc() = "Contains correlation methods.";
	init_correlations(m);
}
