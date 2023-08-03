#include "Filters_py.h"

void init_module(py::module &m)
{
	m.def("boxcar",&boxcarFilter_py<double,double>,"Signal"_a.noconvert(),"order"_a);
	m.def("boxcar",&boxcarFilter_py<float,float>,"Signal"_a.noconvert(),"order"_a);
	//m.def("boxcar",&boxcarFilter_py<uint8_t,uint16_t>,"Signal"_a.noconvert(),"order"_a);
	
	m.def("boxcarAVX",&boxcarFilterAVX_py<double,double>,"Signal"_a.noconvert(),"order"_a);
	m.def("boxcarAVX",&boxcarFilterAVX_py<float,float>,"Signal"_a.noconvert(),"order"_a);
	
	m.def("filterAVX",&customFilterAVX_py<double,double>,"Signal"_a.noconvert(),"filter"_a.noconvert());
	m.def("filterAVX",&customFilterAVX_py<float,float>,"Signal"_a.noconvert(),"filter"_a.noconvert());
}

PYBIND11_MODULE(libfilters, m)
{
	m.doc() = "Might work, might not.";
	init_module(m);
}
