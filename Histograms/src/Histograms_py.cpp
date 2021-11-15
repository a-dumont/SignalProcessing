#include "Histograms_py.h"

void init_histograms(py::module &m)
{
	m.def("histogram",&Histogram_py<np_double,int>, "x"_a,"bins"_a,"A function that generates an histogram based on the number of bins.");
	m.def("histogram",&Histogram_Density_py<np_double,bool>, "x"_a,"bins"_a,"density"_a,"A function that generates an histogram based on the number of bins.");
	m.def("histogram",&Histogram_py<np_double>, "x"_a,"bins"_a,"A function that generates an histogram based on pre determined bins.");
	m.def("histogram",&Histogram_Density_py<np_double>, "x"_a,"bins"_a,"density"_a,"A function that generates an histogram based on pre determined bins.");
	m.def("histogram2d",&Histogram_2D_py<np_double>, "x"_a,"y"_a,"bins"_a);
	m.def("histogram2d",&Histogram_2D_py<np_double,np_double>, "x"_a,"y"_a,"bins"_a);
	m.def("histogram2d",&Histogram_2D_Density_py<np_double>,"x"_a,"y"_a,"bins"_a,"density"_a);
	m.def("histogram2d",&Histogram_2D_Density_py<np_double,np_double>,"x"_a,"y"_a,"bins"_a,"density"_a);
	m.def("find_first_in_bin",&Find_First_In_Bin_py<np_double,np_double>,"data"_a,"edges"_a);
	m.def("find_first_in_bin_2d",&Find_First_In_Bin_2D_py<np_double,np_double>,"xdata"_a,"ydata"_a,"xedges"_a,"yedges"_a);
	m.def("histogram_displacement2d",&Histogram_And_Displacement_2D_py<np_double>, "x"_a,"y"_a,"bins"_a);
	m.def("histogram_displacement2d",&Histogram_And_Displacement_2D_py<np_double,np_double>, "x"_a,"y"_a,"bins"_a);
	py::class_<cHistogram2D_py<np_double>>(m,"Histogram2D")
			.def(py::init<np_double,np_double,int>())
			.def("getHistogram",&cHistogram2D_py<np_double>::getHistogram)
			.def("getEdges",&cHistogram2D_py<np_double>::getEdges)
			.def("accumulate",&cHistogram2D_py<np_double>::accumulate)
			.def("getCount",&cHistogram2D_py<np_double>::getCount)
			.def("getNbins",&cHistogram2D_py<np_double>::getNbins)
			.def("setEdges",&cHistogram2D_py<np_double>::setEdges)
			.def("resetHistogram",&cHistogram2D_py<np_double>::resetHistogram);
	py::class_<cHistogram_2D_Density_py<np_double>>(m,"HistogramDensity2D")
			.def(py::init<np_double,np_double,int>())
			.def("getHistogram",&cHistogram_2D_Density_py<np_double>::getHistogram)
			.def("getEdges",&cHistogram_2D_Density_py<np_double>::getEdges)
			.def("accumulate",&cHistogram_2D_Density_py<np_double>::accumulate)
			.def("getCount",&cHistogram_2D_Density_py<np_double>::getCount)
			.def("getNbins",&cHistogram_2D_Density_py<np_double>::getNbins)	
			.def("setEdges",&cHistogram_2D_Density_py<np_double>::setEdges)
			.def("resetHistogram",&cHistogram_2D_Density_py<np_double>::resetHistogram);

	py::class_<cHistogram_And_Displacement_2D_py<np_double>>(m,"HistogramAndDisplacement2D")
			.def(py::init<np_double,np_double,int>())
			.def("getHistogram",&cHistogram_And_Displacement_2D_py<np_double>::getHistogram)
			.def("getEdges",&cHistogram_And_Displacement_2D_py<np_double>::getEdges)
			.def("accumulate",&cHistogram_And_Displacement_2D_py<np_double>::accumulate)
			.def("getCount",&cHistogram_And_Displacement_2D_py<np_double>::getCount)
			.def("getNbins",&cHistogram_And_Displacement_2D_py<np_double>::getNbins)
			.def("setEdges",&cHistogram_And_Displacement_2D_py<np_double>::setEdges)
			.def("resetHistogram",&cHistogram_And_Displacement_2D_py<np_double>::resetHistogram);
}

PYBIND11_MODULE(libhistograms, m)
{
	m.doc() = "Computes Histograms and is meant to be a replacement for numpy in some simple scenarios.";
	init_histograms(m);
}
